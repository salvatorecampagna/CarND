#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <cppad/cppad.hpp>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

using CppAD::AD;

// Uses the car's kinematic model equations to compute the state
// of the car after dt seconds. So given the state vector (x, y, psi, v, cte and epsi)
// at time t, returns the state of the car at time t + dt 
class StateEstimator
{
public:
  static Eigen::VectorXd nextState(Eigen::VectorXd state, double delta, double a, double dt, double Lf)
  {
    // Current state
    double x = state[0];
    double y = state[1];
    double psi = state[2];
    double v = state[3];
    double cte = state[4];
    double epsi = state[5];

    // Next state estimation using Kinematic Model (car coordinate system)
    Eigen::VectorXd state_next(6);
    state_next(0) = x + v * dt;
    state_next(1) = y + v * sin(psi) * dt;
    state_next(2) = psi - v * delta / Lf * dt;
    state_next(3) = v + a * dt;
    state_next(4) = cte + v * sin(epsi) * dt;
    state_next(5) = epsi - v * delta / Lf * dt;

    return state_next;
  }
};

// For convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
          double delta = j[1]["steering_angle"];
          double a = j[1]["throttle"];

          // Convert x and y points from map's coordinate system
          // to car's coordinate system using translation and
          // rotation equations
          for (size_t i = 0; i < ptsx.size(); i++)
          {
            double dx = ptsx[i] - px;
            double dy = ptsy[i] - py;

            ptsx[i] = (dx * cos(0 - psi) - dy * sin(0 - psi));
            ptsy[i] = (dx * sin(0 - psi) + dy * cos(0 - psi));
          }

          double *ptrx = &ptsx[0];
          double *ptry = &ptsy[0];
          Eigen::Map<Eigen::VectorXd> ptsx_transform(ptrx, 6);
          Eigen::Map<Eigen::VectorXd> ptsy_transform(ptry, 6);

          // Get the coefficients of a 3rd order polynomial fitting
          // the reference line
          auto coeffs = polyfit(ptsx_transform, ptsy_transform, 3);

          // Compute cte and epsi
          double cte = polyeval(coeffs, 0);
          double epsi = -atan(coeffs[1]);


          // State vector (x, y, psi, v, cte, epsi)
          Eigen::VectorXd state(6);
          // x = 0, y = 0, psi = 0 in car reference system
          // see equations above for conversion from map to
          // car coordinate system
          state << 0, 0, 0, v, cte, epsi;

          double Lf = 2.67;

          // Once a control input is computed we have to consider the delay taken
          // for the command to be actually executed. If we don't take into account
          // the delay the command will be executed when the car is in a future
          // state (say 100 ms in future with respect to when it is computed).
          // This would have negative effects on the car control.
          // The idea, then, is to use the kinematic model of the car to estimate
          // the state of the car 100 ms in the future and use that state as the
          // state for computing the control commands (steering angle and acceleration)
          state = StateEstimator::nextState(state, delta, a, 0.1, Lf);

          // Use the estimated state (100 ms in the future) to calculate the control
          // commands. vars[0]: steering angle, vars[1]: acceleration
          auto vars = mpc.Solve(state, coeffs);

          json msgJson;
          // Divide by deg2rad(25) before sending the steering value back,
          // otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          
          // Adjust the steering sign. In the simulator however, a positive value implies a right
          // turn and a negative value implies a left turn
          msgJson["steering_angle"] = - vars[0] / (deg2rad(25) * Lf);
          msgJson["throttle"] = vars[1];

          // Display the MPC predicted trajectory 
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          // Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
