// Reenvía /clock a `frecuencia_hz` de tiempo simulado en /clock_lento.
//
// Gazebo publica /clock en cada paso de física (1000 Hz). Cada nodo con
// use_sim_time lo recibe entero: en Python eso cuesta ~40 % de un núcleo por
// nodo aunque el nodo no haga nada. Nuestros temporizadores van a 10 Hz, así
// que un reloj de 100 Hz alcanza y se procesa acá, en C++, una sola vez.

#include <cstdint>
#include <limits>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "rosgraph_msgs/msg/clock.hpp"

class RelojLento : public rclcpp::Node
{
public:
  RelojLento()
  : Node("reloj_lento")
  {
    const double frecuencia = declare_parameter("frecuencia_hz", 100.0);
    periodo_ns_ = frecuencia > 0.0 ? static_cast<int64_t>(1e9 / frecuencia) : 0;
    // Confiable: lo aceptan tanto suscriptores best effort como reliable.
    publicador_ = create_publisher<rosgraph_msgs::msg::Clock>(
      "clock_lento", rclcpp::QoS(1).reliable());
    suscripcion_ = create_subscription<rosgraph_msgs::msg::Clock>(
      "/clock", rclcpp::ClockQoS(),
      [this](rosgraph_msgs::msg::Clock::ConstSharedPtr msg) {reenviar(*msg);});
  }

private:
  void reenviar(const rosgraph_msgs::msg::Clock & msg)
  {
    const int64_t ahora = rclcpp::Time(msg.clock).nanoseconds();
    // Un salto hacia atrás es un reinicio del mundo: reenviar enseguida.
    if (ahora < ultimo_ns_ || ahora - ultimo_ns_ >= periodo_ns_) {
      ultimo_ns_ = ahora;
      publicador_->publish(msg);
    }
  }

  int64_t periodo_ns_{0};
  // Lejos en el pasado para que el primer mensaje siempre pase.
  int64_t ultimo_ns_{std::numeric_limits<int64_t>::min() / 2};
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr publicador_;
  rclcpp::Subscription<rosgraph_msgs::msg::Clock>::SharedPtr suscripcion_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RelojLento>());
  rclcpp::shutdown();
  return 0;
}
