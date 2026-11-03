#!/usr/bin/env python3
"""Resumen de una corrida y sugerencias de params/ para la siguiente.

Uso:
    python3 plan_de_trabajo/analizar_corrida.py <carpeta_registro> [launch.log]

<carpeta_registro> es la que crea el nodo `registro` (~/.ros/polibots/<fecha-hora>).
launch.log es el de ~/.ros/log/<fecha>-<host>-<pid>/launch.log de esa corrida.
Las reglas salen de plan_de_trabajo/LOGS_CLASIFICACION.md (sección 5). Son sugerencias:
cambiar pocos parámetros y solo si el dato lo justifica.
"""
import csv
import os
import re
import sys

import numpy as np


def leer(carpeta, nombre):
    ruta = os.path.join(carpeta, nombre + '.csv')
    if not os.path.exists(ruta):
        return []
    with open(ruta) as f:
        return list(csv.DictReader(f))


def numeros(filas, campo):
    valores = []
    for fila in filas:
        try:
            valores.append(float(fila[campo]))
        except (KeyError, TypeError, ValueError):
            pass
    return np.array(valores)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    carpeta = sys.argv[1]
    log = open(sys.argv[2], errors='replace').read() if len(sys.argv) > 2 else ''
    sugerencias = []

    print(f'== Registro: {carpeta}')

    # Color de los pixeles rojizos (calibración HSV).
    rojo = [f for f in leer(carpeta, 'rojo') if f.get('s5')]
    if rojo:
        s5, v5 = numeros(rojo, 's5'), numeros(rojo, 'v5')
        h5, h95 = numeros(rojo, 'h5'), numeros(rojo, 'h95')
        n = numeros(rojo, 'pixeles_rojizos')
        con_caja = n > np.percentile(n, 75)   # cuadros con más rojo: la caja a la vista
        print(f'Rojo: {len(rojo)} cuadros con pixeles rojizos; en los que más rojo tienen, '
              f'S p5 mediana {np.median(s5[con_caja]):.0f}, V p5 mediana '
              f'{np.median(v5[con_caja]):.0f}, H de {np.median(h5[con_caja]):.0f} a '
              f'{np.median(h95[con_caja]):.0f} (negativo = del lado de 170-180)')
        s_min, v_min = np.median(s5[con_caja]) - 10, np.median(v5[con_caja]) - 10
        if s_min < 120:
            sugerencias.append(f'detector_scan.saturacion_min: 120 -> {max(60, s_min):.0f}')
        if v_min < 80:
            sugerencias.append(f'detector_scan.valor_min: 80 -> {max(40, v_min):.0f}')
    else:
        print('Rojo: no hubo cuadros con pixeles rojizos (¿la cámara anduvo?)')

    # Localización.
    pose = leer(carpeta, 'pose')
    if pose:
        sxy = numeros(pose, 'sigma_xy')
        print(f'amcl: sigma xy mediana {np.median(sxy):.3f} m, p95 {np.percentile(sxy, 95):.3f} m, '
              f'máx {sxy.max():.3f} m')
        if np.percentile(sxy, 95) > 0.25:   # la covarianza de amcl es pesimista (~4x el error real)
            sugerencias.append('amcl poco preciso: revisar el arranque (tiempo_localizacion_s 5 -> 8)')

    # Movimiento: zona muerta con velocidades bajas.
    mov = leer(carpeta, 'movimiento')
    if mov:
        cvx, cvy, cwz = (numeros(mov, k) for k in ('cmd_vx', 'cmd_vy', 'cmd_wz'))
        ovx, ovy, owz = (numeros(mov, k) for k in ('odom_vx', 'odom_vy', 'odom_wz'))
        lento = (np.hypot(cvx, cvy) > 0.04) & (np.hypot(cvx, cvy) < 0.15)
        if lento.sum() > 10:
            rinde = np.median(np.hypot(ovx, ovy)[lento] / np.hypot(cvx, cvy)[lento])
            print(f'Traslación lenta (0.04-0.15 m/s pedidos): rinde {100 * rinde:.0f}% '
                  f'({lento.sum()} muestras)')
            if rinde < 0.4:
                sugerencias.append('zona muerta en traslación: min_speed_xy 0.08 -> 0.12, '
                                   'velocidad_destrabe 0.15 -> 0.2')
        giro = (np.abs(cwz) > 0.2) & (np.abs(cwz) < 0.6) & (np.hypot(cvx, cvy) < 0.05)
        if giro.sum() > 10:
            rinde = np.median(np.abs(owz[giro]) / np.abs(cwz[giro]))
            print(f'Giro en el lugar (0.2-0.6 rad/s pedidos): rinde {100 * rinde:.0f}% '
                  f'({giro.sum()} muestras)')
            if rinde < 0.4:
                sugerencias.append('zona muerta en giro: min_speed_theta 0.3 -> 0.45, '
                                   'giro_minimo_inspeccion 0.0 -> 0.4')
        rapido = np.hypot(ovx, ovy)
        p50 = np.median(rapido[rapido > 0.05]) if (rapido > 0.05).any() else 0.0
        print(f'Velocidad medida: p50 {p50:.2f} m/s, p95 {np.percentile(rapido, 95):.2f} m/s')

    # Roces.
    roces = leer(carpeta, 'roces')
    if roces:
        d = numeros(roces, 'distancia_min')
        episodios, adentro = 0, False
        for v in d:
            if v < 0.02 and not adentro:
                episodios, adentro = episodios + 1, True
            elif v > 0.04:
                adentro = False
        print(f'Roces: distancia mínima {d.min():.3f} m; episodios < 2 cm: {episodios}')
        if episodios >= 2:
            sugerencias.append('roces: footprint_padding del costmap local 0.03 -> 0.05 '
                               'o max_vel_x/y 0.5 -> 0.35')

    # CPU.
    cpu = leer(carpeta, 'cpu')
    if cpu:
        hz = numeros(cpu, 'cmd_vel_hz')
        carga = numeros(cpu, 'carga_1min')
        hz_med = np.median(hz[hz > 0]) if (hz > 0).any() else 0.0
        print(f'CPU: carga mediana {np.median(carga):.1f}; /cmd_vel {hz_med:.1f} Hz')

    # Log de launch: eventos de la misión.
    if log:
        cuenta = {
            'control lento (missed rate)': len(re.findall(r'missed its desired rate', log)),
            'sin trayectorias válidas': len(re.findall(r'No valid trajectories', log)),
            'arranque en zona letal': len(re.findall(r'lethal space', log)),
            'atascos (subobjetivo)': len(re.findall(r'Atascado en', log)),
            'abandonos (sin desplazarse)': len(re.findall(r'sin desplazarse', log)),
            'cortes por recovery': len(re.findall(r'entró en recovery', log)),
            'inspecciones': len(re.findall(r'Inspeccionando candidato', log)),
            'candidatos no rojos': len(re.findall(r'no es rojo', log)),
            'reportes': len(re.findall(r'Publicada víctima', log)),
        }
        print('Eventos:', ', '.join(f'{k} {v}' for k, v in cuenta.items()))
        for linea in re.findall(r'\[ *[0-9.]+ s\] [A-Z_]+ -> [A-Z_]+', log):
            print('  ', linea)
        for linea in re.findall(r'Localizado \([^)]*\)[^\n]*', log)[:1]:
            print('  ', linea)
        if cuenta['control lento (missed rate)'] > 150:
            sugerencias.append('CPU justa: vx/vy/vtheta_samples 15/11/9 -> 9/7/7')
        if cuenta['candidatos no rojos'] >= 6:
            sugerencias.append('muchos candidatos falsos: despeje_candidato_claro 0.2 -> 0.25 '
                               'o detector_no_mapeados.margen_pared 0.15 -> 0.20')
        if cuenta['abandonos (sin desplazarse)'] + cuenta['atascos (subobjetivo)'] >= 4:
            sugerencias.append('se traba seguido: tiempo_atasco_s 4 -> 3, '
                               'distancia_subobjetivo 0.5 -> 0.6')

    print('\n== Sugerencias (cambiar pocas, solo con el dato a la vista):')
    print('\n'.join('  - ' + s for s in sugerencias) or '  (ninguna: dejar todo igual)')
    print('  - Siempre, en la arena real: juntar_victimas.mitad_caja = lado de la caja / 2 '
          '(cajas de 18 cm: 0.09)')


if __name__ == '__main__':
    main()
