#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <math.h>

#include "json.hpp"

#include "map.h"
#include "vehicle.h"
#include "road.h"
#include "planner.h"
#include "params.h"

// for convenience
using json = nlohmann::json;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s)
{
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != std::string::npos)
  {
    return "";
  }
  else if (b1 != std::string::npos && b2 != std::string::npos)
  {
    return s.substr(b1, b2 - b1 + 2);
  }

  return "";
}

int main(int argc, char *argv[])
{
  uWS::Hub h;
  std::fstream telemetry_log;
  bool enable_log_telemetry = false;

  if(argc > 1)
  {
    enable_log_telemetry = true;
    telemetry_log.open(argv[1], std::fstream::out);
    if (!telemetry_log.is_open())
    {
      std::cerr << "Unable to open telemetry log file: " << argv[1] << std::endl;
      return 0;
    }
  }

  std::string map_file_ = "../data/highway_map.csv";
  int counter = 0;

  // Map & Road & Vehicle & Planner instances
  Map map (map_file_);
  Road road;
  Vehicle car;
  Planner planner;

  h.onMessage([
    &map,
    &road,
    &car,
    &planner,
    &counter,
    &telemetry_log,
    &enable_log_telemetry](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {
      auto s = hasData(data);
      if (s != "")
      {
        auto j = json::parse(s);

        std::string event = j[0].get<std::string>();
        if (event == "telemetry")
        {
          // j[1] is the data JSON object
          counter++;

          // Car data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          if (car_d < 0.5 || car_d > 11.5)
          {
            std::cerr << "#Lane violation. d: " << car_d << std::endl;
          }

          if (car_speed > 50)
          {
            std::cerr << "#Speed violation: v: " << car_speed << std::endl;
          }

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];

          json msgJson;

          std::vector<double> next_x_vals;
          std::vector<double> next_y_vals;

          if (enable_log_telemetry)
          {
            telemetry_log << counter << ", ";
            telemetry_log << car_s << ", ";
            telemetry_log << car_d << ", ";
            telemetry_log << car_speed << std::endl;
            if (car_d < 0.5 || car_d > 11.5)
            {
              telemetry_log << "#Lane violation. d: " << car_d << std::endl;
            }

            if (car_speed > 50)
            {
              telemetry_log << "#Speed violation: v: " << car_speed << std::endl;
            }
            telemetry_log.flush();
          }

          car.update_vehicle_status(car_x, car_y, car_speed, car_s, car_d, car_yaw);

          std::vector<Vehicle> left_lane;
          std::vector<Vehicle> center_lane;
          std::vector<Vehicle> right_lane;

          for (int i = 0; i < sensor_fusion.size(); ++i)
          {
            // Get sensor fusion data
            int id = sensor_fusion[i][0];
            double x = sensor_fusion[i][1];
            double y = sensor_fusion[i][2];
            double vx = sensor_fusion[i][3];
            double vy = sensor_fusion[i][4];
            double v = sqrt(vx*vx + vy*vy);
            double s = sensor_fusion[i][5];
            double d = sensor_fusion[i][6];

            // Get i-th vehicle data
            Vehicle vehicle(id, x, y, v, s, d);
            unsigned int vehicle_lane = vehicle.lane();

            // Put the i-th vehicle in the corresponding lane
            if (vehicle_lane == Lane::LEFT_LANE)
            {
              left_lane.push_back(vehicle);
            }
            else if (vehicle_lane == Lane::CENTER_LANE)
            {
              center_lane.push_back(vehicle);
            }
            else if (vehicle_lane == Lane::RIGHT_LANE)
            {
              right_lane.push_back(vehicle);
            }
            else
            {
              std::cerr << "unknown" << std::endl;
            }
          }

          // Update road status, eahc vehicle in its lane
          road.update_lanes(left_lane, center_lane, right_lane);

          int n = previous_path_x.size();
          for(int i = 0; i < n; i++)
          {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          std::vector<std::vector<double>> trajectory;
          trajectory.push_back(next_x_vals);
          trajectory.push_back(next_y_vals);
          planner.new_trajectory(map, road, car, trajectory);

          msgJson["next_x"] = trajectory[0];
          msgJson["next_y"] = trajectory[1];

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);

        }
      }
      else
      {
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
                     size_t, size_t)
  {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1)
    {
      res->end(s.data(), s.length());
    }
    else
    {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req)
  {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length)
  {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }
  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
