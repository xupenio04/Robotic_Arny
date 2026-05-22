// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from omx_interfaces:srv/ExecuteTrajectory.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "omx_interfaces/srv/execute_trajectory.hpp"


#ifndef OMX_INTERFACES__SRV__DETAIL__EXECUTE_TRAJECTORY__BUILDER_HPP_
#define OMX_INTERFACES__SRV__DETAIL__EXECUTE_TRAJECTORY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "omx_interfaces/srv/detail/execute_trajectory__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace omx_interfaces
{

namespace srv
{

namespace builder
{

class Init_ExecuteTrajectory_Request_trajectory
{
public:
  Init_ExecuteTrajectory_Request_trajectory()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::omx_interfaces::srv::ExecuteTrajectory_Request trajectory(::omx_interfaces::srv::ExecuteTrajectory_Request::_trajectory_type arg)
  {
    msg_.trajectory = std::move(arg);
    return std::move(msg_);
  }

private:
  ::omx_interfaces::srv::ExecuteTrajectory_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::omx_interfaces::srv::ExecuteTrajectory_Request>()
{
  return omx_interfaces::srv::builder::Init_ExecuteTrajectory_Request_trajectory();
}

}  // namespace omx_interfaces


namespace omx_interfaces
{

namespace srv
{

namespace builder
{

class Init_ExecuteTrajectory_Response_message
{
public:
  explicit Init_ExecuteTrajectory_Response_message(::omx_interfaces::srv::ExecuteTrajectory_Response & msg)
  : msg_(msg)
  {}
  ::omx_interfaces::srv::ExecuteTrajectory_Response message(::omx_interfaces::srv::ExecuteTrajectory_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::omx_interfaces::srv::ExecuteTrajectory_Response msg_;
};

class Init_ExecuteTrajectory_Response_success
{
public:
  Init_ExecuteTrajectory_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ExecuteTrajectory_Response_message success(::omx_interfaces::srv::ExecuteTrajectory_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_ExecuteTrajectory_Response_message(msg_);
  }

private:
  ::omx_interfaces::srv::ExecuteTrajectory_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::omx_interfaces::srv::ExecuteTrajectory_Response>()
{
  return omx_interfaces::srv::builder::Init_ExecuteTrajectory_Response_success();
}

}  // namespace omx_interfaces


namespace omx_interfaces
{

namespace srv
{

namespace builder
{

class Init_ExecuteTrajectory_Event_response
{
public:
  explicit Init_ExecuteTrajectory_Event_response(::omx_interfaces::srv::ExecuteTrajectory_Event & msg)
  : msg_(msg)
  {}
  ::omx_interfaces::srv::ExecuteTrajectory_Event response(::omx_interfaces::srv::ExecuteTrajectory_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::omx_interfaces::srv::ExecuteTrajectory_Event msg_;
};

class Init_ExecuteTrajectory_Event_request
{
public:
  explicit Init_ExecuteTrajectory_Event_request(::omx_interfaces::srv::ExecuteTrajectory_Event & msg)
  : msg_(msg)
  {}
  Init_ExecuteTrajectory_Event_response request(::omx_interfaces::srv::ExecuteTrajectory_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_ExecuteTrajectory_Event_response(msg_);
  }

private:
  ::omx_interfaces::srv::ExecuteTrajectory_Event msg_;
};

class Init_ExecuteTrajectory_Event_info
{
public:
  Init_ExecuteTrajectory_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ExecuteTrajectory_Event_request info(::omx_interfaces::srv::ExecuteTrajectory_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_ExecuteTrajectory_Event_request(msg_);
  }

private:
  ::omx_interfaces::srv::ExecuteTrajectory_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::omx_interfaces::srv::ExecuteTrajectory_Event>()
{
  return omx_interfaces::srv::builder::Init_ExecuteTrajectory_Event_info();
}

}  // namespace omx_interfaces

#endif  // OMX_INTERFACES__SRV__DETAIL__EXECUTE_TRAJECTORY__BUILDER_HPP_
