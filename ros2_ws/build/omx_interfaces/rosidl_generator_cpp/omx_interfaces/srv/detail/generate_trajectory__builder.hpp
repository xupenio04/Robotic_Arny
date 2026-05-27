// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from omx_interfaces:srv/GenerateTrajectory.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "omx_interfaces/srv/generate_trajectory.hpp"


#ifndef OMX_INTERFACES__SRV__DETAIL__GENERATE_TRAJECTORY__BUILDER_HPP_
#define OMX_INTERFACES__SRV__DETAIL__GENERATE_TRAJECTORY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "omx_interfaces/srv/detail/generate_trajectory__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace omx_interfaces
{

namespace srv
{

namespace builder
{

class Init_GenerateTrajectory_Request_ts
{
public:
  explicit Init_GenerateTrajectory_Request_ts(::omx_interfaces::srv::GenerateTrajectory_Request & msg)
  : msg_(msg)
  {}
  ::omx_interfaces::srv::GenerateTrajectory_Request ts(::omx_interfaces::srv::GenerateTrajectory_Request::_ts_type arg)
  {
    msg_.ts = std::move(arg);
    return std::move(msg_);
  }

private:
  ::omx_interfaces::srv::GenerateTrajectory_Request msg_;
};

class Init_GenerateTrajectory_Request_qf
{
public:
  explicit Init_GenerateTrajectory_Request_qf(::omx_interfaces::srv::GenerateTrajectory_Request & msg)
  : msg_(msg)
  {}
  Init_GenerateTrajectory_Request_ts qf(::omx_interfaces::srv::GenerateTrajectory_Request::_qf_type arg)
  {
    msg_.qf = std::move(arg);
    return Init_GenerateTrajectory_Request_ts(msg_);
  }

private:
  ::omx_interfaces::srv::GenerateTrajectory_Request msg_;
};

class Init_GenerateTrajectory_Request_qi
{
public:
  Init_GenerateTrajectory_Request_qi()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GenerateTrajectory_Request_qf qi(::omx_interfaces::srv::GenerateTrajectory_Request::_qi_type arg)
  {
    msg_.qi = std::move(arg);
    return Init_GenerateTrajectory_Request_qf(msg_);
  }

private:
  ::omx_interfaces::srv::GenerateTrajectory_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::omx_interfaces::srv::GenerateTrajectory_Request>()
{
  return omx_interfaces::srv::builder::Init_GenerateTrajectory_Request_qi();
}

}  // namespace omx_interfaces


namespace omx_interfaces
{

namespace srv
{

namespace builder
{

class Init_GenerateTrajectory_Response_message
{
public:
  explicit Init_GenerateTrajectory_Response_message(::omx_interfaces::srv::GenerateTrajectory_Response & msg)
  : msg_(msg)
  {}
  ::omx_interfaces::srv::GenerateTrajectory_Response message(::omx_interfaces::srv::GenerateTrajectory_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::omx_interfaces::srv::GenerateTrajectory_Response msg_;
};

class Init_GenerateTrajectory_Response_success
{
public:
  explicit Init_GenerateTrajectory_Response_success(::omx_interfaces::srv::GenerateTrajectory_Response & msg)
  : msg_(msg)
  {}
  Init_GenerateTrajectory_Response_message success(::omx_interfaces::srv::GenerateTrajectory_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_GenerateTrajectory_Response_message(msg_);
  }

private:
  ::omx_interfaces::srv::GenerateTrajectory_Response msg_;
};

class Init_GenerateTrajectory_Response_trajectory
{
public:
  Init_GenerateTrajectory_Response_trajectory()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GenerateTrajectory_Response_success trajectory(::omx_interfaces::srv::GenerateTrajectory_Response::_trajectory_type arg)
  {
    msg_.trajectory = std::move(arg);
    return Init_GenerateTrajectory_Response_success(msg_);
  }

private:
  ::omx_interfaces::srv::GenerateTrajectory_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::omx_interfaces::srv::GenerateTrajectory_Response>()
{
  return omx_interfaces::srv::builder::Init_GenerateTrajectory_Response_trajectory();
}

}  // namespace omx_interfaces


namespace omx_interfaces
{

namespace srv
{

namespace builder
{

class Init_GenerateTrajectory_Event_response
{
public:
  explicit Init_GenerateTrajectory_Event_response(::omx_interfaces::srv::GenerateTrajectory_Event & msg)
  : msg_(msg)
  {}
  ::omx_interfaces::srv::GenerateTrajectory_Event response(::omx_interfaces::srv::GenerateTrajectory_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::omx_interfaces::srv::GenerateTrajectory_Event msg_;
};

class Init_GenerateTrajectory_Event_request
{
public:
  explicit Init_GenerateTrajectory_Event_request(::omx_interfaces::srv::GenerateTrajectory_Event & msg)
  : msg_(msg)
  {}
  Init_GenerateTrajectory_Event_response request(::omx_interfaces::srv::GenerateTrajectory_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_GenerateTrajectory_Event_response(msg_);
  }

private:
  ::omx_interfaces::srv::GenerateTrajectory_Event msg_;
};

class Init_GenerateTrajectory_Event_info
{
public:
  Init_GenerateTrajectory_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GenerateTrajectory_Event_request info(::omx_interfaces::srv::GenerateTrajectory_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_GenerateTrajectory_Event_request(msg_);
  }

private:
  ::omx_interfaces::srv::GenerateTrajectory_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::omx_interfaces::srv::GenerateTrajectory_Event>()
{
  return omx_interfaces::srv::builder::Init_GenerateTrajectory_Event_info();
}

}  // namespace omx_interfaces

#endif  // OMX_INTERFACES__SRV__DETAIL__GENERATE_TRAJECTORY__BUILDER_HPP_
