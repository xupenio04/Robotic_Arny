// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from omx_interfaces:srv/SetGripper.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "omx_interfaces/srv/set_gripper.h"


#ifndef OMX_INTERFACES__SRV__DETAIL__SET_GRIPPER__STRUCT_H_
#define OMX_INTERFACES__SRV__DETAIL__SET_GRIPPER__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/SetGripper in the package omx_interfaces.
typedef struct omx_interfaces__srv__SetGripper_Request
{
  double position;
} omx_interfaces__srv__SetGripper_Request;

// Struct for a sequence of omx_interfaces__srv__SetGripper_Request.
typedef struct omx_interfaces__srv__SetGripper_Request__Sequence
{
  omx_interfaces__srv__SetGripper_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} omx_interfaces__srv__SetGripper_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'message'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/SetGripper in the package omx_interfaces.
typedef struct omx_interfaces__srv__SetGripper_Response
{
  bool success;
  rosidl_runtime_c__String message;
} omx_interfaces__srv__SetGripper_Response;

// Struct for a sequence of omx_interfaces__srv__SetGripper_Response.
typedef struct omx_interfaces__srv__SetGripper_Response__Sequence
{
  omx_interfaces__srv__SetGripper_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} omx_interfaces__srv__SetGripper_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  omx_interfaces__srv__SetGripper_Event__request__MAX_SIZE = 1
};
// response
enum
{
  omx_interfaces__srv__SetGripper_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/SetGripper in the package omx_interfaces.
typedef struct omx_interfaces__srv__SetGripper_Event
{
  service_msgs__msg__ServiceEventInfo info;
  omx_interfaces__srv__SetGripper_Request__Sequence request;
  omx_interfaces__srv__SetGripper_Response__Sequence response;
} omx_interfaces__srv__SetGripper_Event;

// Struct for a sequence of omx_interfaces__srv__SetGripper_Event.
typedef struct omx_interfaces__srv__SetGripper_Event__Sequence
{
  omx_interfaces__srv__SetGripper_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} omx_interfaces__srv__SetGripper_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // OMX_INTERFACES__SRV__DETAIL__SET_GRIPPER__STRUCT_H_
