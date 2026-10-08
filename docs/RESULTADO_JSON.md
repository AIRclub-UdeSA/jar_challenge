# `resultado.json`: contrato estable del resultado de una corrida

> Contrato (issue #4). Lo emite `tools/juez.py --salida` (y `juez_mock.py --salida`
> en versión reducida). El leaderboard (`ranking.py`, Pages) y el CI solo pueden
> asumir estos campos. Agregar un campo es compatible; quitar o renombrar uno
> rompe el contrato y exige nueva versión.
>
> `x`/`y` de reportes y eventos pueden ser `null`: un reporte mal formado con
> valores no finitos (§7.2) se sanea a `null` antes de escribir (`sanear_json`
> en `juez_mock.py`), porque `NaN` no es JSON válido y rompería `JSON.parse`
> en Pages.

## Campos base (también los emite `juez_mock.py`)

| Campo | Tipo | Origen |
|---|---|---|
| `termino_por` | `done` \| `tiempo` \| `salida` \| `vuelco` \| `juez` \| `inmovil` \| `null` | motivo de fin de corrida |
| `mapa_publicado` | bool | si `/map` llegó a publicarse |
| `reportes` | `[{t, x, y}]` (`t` = s desde `/map`, `null` si fue antes; `x`/`y` = `null` si no finitos) | un elemento por mensaje en `/victimas` |
| `reportes_mal_formados` | `[{t, frame_id}]` | subconjunto de reportes con `frame_id != map` o no finitos |
| `antes_del_mapa` | `{cmd_vel_no_nulo: int, reportes: int}` | actividad prohibida previa a `/map` |
| `violaciones` | `[str]` | lista legible (movimiento/reporte prematuro, QoS, mal formados); `[]` = limpio |

## Campos agregados por `juez.py` (puntaje)

| Campo | Tipo | Origen |
|---|---|---|
| `puntaje` | objeto `Resultado` (ver abajo) | `puntaje.puntuar()` |
| `resumen` | str, p. ej. `total 250 \| víctimas [V1:100 V2:50]` | `Resultado.resumen()` |
| `victimas` | `[[x, y], ...]` en metros, marco `map` | `--mundo` o `--victimas-json` (N = len) |
| `eventos` | `[{t, evento, ...}]` en orden cronológico | registro interno (`--registro` lo duplica en JSONL) |
| `equipo` | str \| null | `--equipo` (si falta, `ranking.py` lo saca del nombre de carpeta) |
| `duracion_s` | float \| null | s desde `/map` hasta ahora |
| `tiempo_total_s` | float \| null | s desde `/map` hasta el fin (desempate §8.2) |
| `tiempo_ultimo_reporte_que_puntua_s` | float \| null | instante del último reporte que puntuó (solo `ranking.py --desempate ultimo_reporte`) |
| `movimiento_despues_de_done` | int | `/cmd_vel` no nulos tras `/done` (se anota, no se sanciona) |
| `medidas_con_ground_truth` | bool | si el regreso se midió con pose real (solo simulador) |

## Objeto `puntaje` (`puntaje.Resultado`)

| Campo | Tipo |
|---|---|
| `por_victima` | `[{indice, indice_reporte \| null, distancia \| null, puntos}]` |
| `reportes_contados` | int (máximo N) |
| `reportes_ignorados` | int (los que exceden N) |
| `puntos_victimas` | int |
| `bono_rapido` | bool (+200, N reportes que puntúan antes de 300 s) |
| `bono_regreso` | bool (+100, `/done` cerca de largada + ≥1 víctima puntuada) |
| `penalizacion_colisiones` | int (n × 50) |
| `penalizacion_salida` | int (0 o 50; además termina la corrida) |
| `total` | int (mínimo 0; 0 si `anulada`) |
| `anulada` | bool (vuelco, §11.2) |
| `tiempo_ultimo_reporte_que_puntua` | float \| null |

## Lo mínimo que necesita `ranking.py`

`equipo`, `puntaje.total`, `puntaje.penalizacion_colisiones`, `puntaje.anulada`,
`tiempo_total_s` (o `tiempo_ultimo_reporte_que_puntua_s` con `--desempate ultimo_reporte`),
`violaciones`, `mapa_publicado`. Todo lo demás es informativo.

## Ejemplo válido (reducido)

```json
{
  "termino_por": "done",
  "mapa_publicado": true,
  "reportes": [{"t": 12.4, "x": 1.02, "y": -0.48}, {"t": 45.1, "x": 2.51, "y": 0.33}],
  "reportes_mal_formados": [],
  "antes_del_mapa": {"cmd_vel_no_nulo": 0, "reportes": 0},
  "violaciones": [],
  "puntaje": {
    "por_victima": [
      {"indice": 0, "indice_reporte": 0, "distancia": 0.12, "puntos": 100},
      {"indice": 1, "indice_reporte": 1, "distancia": 0.4, "puntos": 50}
    ],
    "reportes_contados": 2, "reportes_ignorados": 0, "puntos_victimas": 150,
    "bono_rapido": false, "bono_regreso": false,
    "penalizacion_colisiones": 0, "penalizacion_salida": 0,
    "total": 150, "anulada": false, "tiempo_ultimo_reporte_que_puntua": 45.1
  },
  "resumen": "total 150 | víctimas [V1:100 V2:50]",
  "victimas": [[1.0, -0.5], [2.4, 0.2]],
  "eventos": [{"t": 12.4, "evento": "reporte", "x": 1.02, "y": -0.48, "valido": true}],
  "equipo": "los_osos",
  "duracion_s": 60.2, "tiempo_total_s": 60.2,
  "tiempo_ultimo_reporte_que_puntua_s": 45.1,
  "movimiento_despues_de_done": 0, "medidas_con_ground_truth": true
}
```
