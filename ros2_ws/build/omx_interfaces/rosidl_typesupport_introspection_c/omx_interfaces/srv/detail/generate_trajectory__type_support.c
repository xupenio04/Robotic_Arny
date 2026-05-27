// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from omx_interfaces:srv/GenerateTrajectory.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "omx_interfaces/srv/detail/generate_trajectory__rosidl_typesupport_introspection_c.h"
#include "omx_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "omx_interfaces/srv/detail/generate_trajectory__functions.h"
#include "omx_interfaces/srv/detail/generate_trajectory__struct.h"


// Include directives for member types
// Member `qi`
// Member `qf`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__GenerateTrajectory_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  omx_interfaces__srv__GenerateTrajectory_Request__init(message_memory);
}

void omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__GenerateTrajectory_Request_fini_function(void * message_memory)
{
  omx_interfaces__srv__GenerateTrajectory_Request__fini(message_memory);
}

size_t omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__size_function__GenerateTrajectory_Request__qi(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__get_const_function__GenerateTrajectory_Request__qi(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__get_function__GenerateTrajectory_Request__qi(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__fetch_function__GenerateTrajectory_Request__qi(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__get_const_function__GenerateTrajectory_Request__qi(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__assign_function__GenerateTrajectory_Request__qi(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__get_function__GenerateTrajectory_Request__qi(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__resize_function__GenerateTrajectory_Request__qi(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

size_t omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__size_function__GenerateTrajectory_Request__qf(
  const void * untyped_member)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return member->size;
}

const void * omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__get_const_function__GenerateTrajectory_Request__qf(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__double__Sequence * member =
    (const rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void * omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__get_function__GenerateTrajectory_Request__qf(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  return &member->data[index];
}

void omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__fetch_function__GenerateTrajectory_Request__qf(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__get_const_function__GenerateTrajectory_Request__qf(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__assign_function__GenerateTrajectory_Request__qf(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__get_function__GenerateTrajectory_Request__qf(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

bool omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__resize_function__GenerateTrajectory_Request__qf(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__double__Sequence * member =
    (rosidl_runtime_c__double__Sequence *)(untyped_member);
  rosidl_runtime_c__double__Sequence__fini(member);
  return rosidl_runtime_c__double__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__GenerateTrajectory_Request_message_member_array[3] = {
  {
    "qi",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(omx_interfaces__srv__GenerateTrajectory_Request, qi),  // bytes offset in struct
    NULL,  // default value
    omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__size_function__GenerateTrajectory_Request__qi,  // size() function pointer
    omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__get_const_function__GenerateTrajectory_Request__qi,  // get_const(index) function pointer
    omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__get_function__GenerateTrajectory_Request__qi,  // get(index) function pointer
    omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__fetch_function__GenerateTrajectory_Request__qi,  // fetch(index, &value) function pointer
    omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__assign_function__GenerateTrajectory_Request__qi,  // assign(index, value) function pointer
    omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__resize_function__GenerateTrajectory_Request__qi  // resize(index) function pointer
  },
  {
    "qf",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(omx_interfaces__srv__GenerateTrajectory_Request, qf),  // bytes offset in struct
    NULL,  // default value
    omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__size_function__GenerateTrajectory_Request__qf,  // size() function pointer
    omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__get_const_function__GenerateTrajectory_Request__qf,  // get_const(index) function pointer
    omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__get_function__GenerateTrajectory_Request__qf,  // get(index) function pointer
    omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__fetch_function__GenerateTrajectory_Request__qf,  // fetch(index, &value) function pointer
    omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__assign_function__GenerateTrajectory_Request__qf,  // assign(index, value) function pointer
    omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__resize_function__GenerateTrajectory_Request__qf  // resize(index) function pointer
  },
  {
    "ts",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(omx_interfaces__srv__GenerateTrajectory_Request, ts),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__GenerateTrajectory_Request_message_members = {
  "omx_interfaces__srv",  // message namespace
  "GenerateTrajectory_Request",  // message name
  3,  // number of fields
  sizeof(omx_interfaces__srv__GenerateTrajectory_Request),
  false,  // has_any_key_member_
  omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__GenerateTrajectory_Request_message_member_array,  // message members
  omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__GenerateTrajectory_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__GenerateTrajectory_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__GenerateTrajectory_Request_message_type_support_handle = {
  0,
  &omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__GenerateTrajectory_Request_message_members,
  get_message_typesupport_handle_function,
  &omx_interfaces__srv__GenerateTrajectory_Request__get_type_hash,
  &omx_interfaces__srv__GenerateTrajectory_Request__get_type_description,
  &omx_interfaces__srv__GenerateTrajectory_Request__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_omx_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, omx_interfaces, srv, GenerateTrajectory_Request)() {
  if (!omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__GenerateTrajectory_Request_message_type_support_handle.typesupport_identifier) {
    omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__GenerateTrajectory_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__GenerateTrajectory_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "omx_interfaces/srv/detail/generate_trajectory__rosidl_typesupport_introspection_c.h"
// already included above
// #include "omx_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "omx_interfaces/srv/detail/generate_trajectory__functions.h"
// already included above
// #include "omx_interfaces/srv/detail/generate_trajectory__struct.h"


// Include directives for member types
// Member `trajectory`
#include "trajectory_msgs/msg/joint_trajectory.h"
// Member `trajectory`
#include "trajectory_msgs/msg/detail/joint_trajectory__rosidl_typesupport_introspection_c.h"
// Member `message`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void omx_interfaces__srv__GenerateTrajectory_Response__rosidl_typesupport_introspection_c__GenerateTrajectory_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  omx_interfaces__srv__GenerateTrajectory_Response__init(message_memory);
}

void omx_interfaces__srv__GenerateTrajectory_Response__rosidl_typesupport_introspection_c__GenerateTrajectory_Response_fini_function(void * message_memory)
{
  omx_interfaces__srv__GenerateTrajectory_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember omx_interfaces__srv__GenerateTrajectory_Response__rosidl_typesupport_introspection_c__GenerateTrajectory_Response_message_member_array[3] = {
  {
    "trajectory",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(omx_interfaces__srv__GenerateTrajectory_Response, trajectory),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "success",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(omx_interfaces__srv__GenerateTrajectory_Response, success),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "message",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(omx_interfaces__srv__GenerateTrajectory_Response, message),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers omx_interfaces__srv__GenerateTrajectory_Response__rosidl_typesupport_introspection_c__GenerateTrajectory_Response_message_members = {
  "omx_interfaces__srv",  // message namespace
  "GenerateTrajectory_Response",  // message name
  3,  // number of fields
  sizeof(omx_interfaces__srv__GenerateTrajectory_Response),
  false,  // has_any_key_member_
  omx_interfaces__srv__GenerateTrajectory_Response__rosidl_typesupport_introspection_c__GenerateTrajectory_Response_message_member_array,  // message members
  omx_interfaces__srv__GenerateTrajectory_Response__rosidl_typesupport_introspection_c__GenerateTrajectory_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  omx_interfaces__srv__GenerateTrajectory_Response__rosidl_typesupport_introspection_c__GenerateTrajectory_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t omx_interfaces__srv__GenerateTrajectory_Response__rosidl_typesupport_introspection_c__GenerateTrajectory_Response_message_type_support_handle = {
  0,
  &omx_interfaces__srv__GenerateTrajectory_Response__rosidl_typesupport_introspection_c__GenerateTrajectory_Response_message_members,
  get_message_typesupport_handle_function,
  &omx_interfaces__srv__GenerateTrajectory_Response__get_type_hash,
  &omx_interfaces__srv__GenerateTrajectory_Response__get_type_description,
  &omx_interfaces__srv__GenerateTrajectory_Response__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_omx_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, omx_interfaces, srv, GenerateTrajectory_Response)() {
  omx_interfaces__srv__GenerateTrajectory_Response__rosidl_typesupport_introspection_c__GenerateTrajectory_Response_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, trajectory_msgs, msg, JointTrajectory)();
  if (!omx_interfaces__srv__GenerateTrajectory_Response__rosidl_typesupport_introspection_c__GenerateTrajectory_Response_message_type_support_handle.typesupport_identifier) {
    omx_interfaces__srv__GenerateTrajectory_Response__rosidl_typesupport_introspection_c__GenerateTrajectory_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &omx_interfaces__srv__GenerateTrajectory_Response__rosidl_typesupport_introspection_c__GenerateTrajectory_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "omx_interfaces/srv/detail/generate_trajectory__rosidl_typesupport_introspection_c.h"
// already included above
// #include "omx_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "omx_interfaces/srv/detail/generate_trajectory__functions.h"
// already included above
// #include "omx_interfaces/srv/detail/generate_trajectory__struct.h"


// Include directives for member types
// Member `info`
#include "service_msgs/msg/service_event_info.h"
// Member `info`
#include "service_msgs/msg/detail/service_event_info__rosidl_typesupport_introspection_c.h"
// Member `request`
// Member `response`
#include "omx_interfaces/srv/generate_trajectory.h"
// Member `request`
// Member `response`
// already included above
// #include "omx_interfaces/srv/detail/generate_trajectory__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__GenerateTrajectory_Event_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  omx_interfaces__srv__GenerateTrajectory_Event__init(message_memory);
}

void omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__GenerateTrajectory_Event_fini_function(void * message_memory)
{
  omx_interfaces__srv__GenerateTrajectory_Event__fini(message_memory);
}

size_t omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__size_function__GenerateTrajectory_Event__request(
  const void * untyped_member)
{
  const omx_interfaces__srv__GenerateTrajectory_Request__Sequence * member =
    (const omx_interfaces__srv__GenerateTrajectory_Request__Sequence *)(untyped_member);
  return member->size;
}

const void * omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__get_const_function__GenerateTrajectory_Event__request(
  const void * untyped_member, size_t index)
{
  const omx_interfaces__srv__GenerateTrajectory_Request__Sequence * member =
    (const omx_interfaces__srv__GenerateTrajectory_Request__Sequence *)(untyped_member);
  return &member->data[index];
}

void * omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__get_function__GenerateTrajectory_Event__request(
  void * untyped_member, size_t index)
{
  omx_interfaces__srv__GenerateTrajectory_Request__Sequence * member =
    (omx_interfaces__srv__GenerateTrajectory_Request__Sequence *)(untyped_member);
  return &member->data[index];
}

void omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__fetch_function__GenerateTrajectory_Event__request(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const omx_interfaces__srv__GenerateTrajectory_Request * item =
    ((const omx_interfaces__srv__GenerateTrajectory_Request *)
    omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__get_const_function__GenerateTrajectory_Event__request(untyped_member, index));
  omx_interfaces__srv__GenerateTrajectory_Request * value =
    (omx_interfaces__srv__GenerateTrajectory_Request *)(untyped_value);
  *value = *item;
}

void omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__assign_function__GenerateTrajectory_Event__request(
  void * untyped_member, size_t index, const void * untyped_value)
{
  omx_interfaces__srv__GenerateTrajectory_Request * item =
    ((omx_interfaces__srv__GenerateTrajectory_Request *)
    omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__get_function__GenerateTrajectory_Event__request(untyped_member, index));
  const omx_interfaces__srv__GenerateTrajectory_Request * value =
    (const omx_interfaces__srv__GenerateTrajectory_Request *)(untyped_value);
  *item = *value;
}

bool omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__resize_function__GenerateTrajectory_Event__request(
  void * untyped_member, size_t size)
{
  omx_interfaces__srv__GenerateTrajectory_Request__Sequence * member =
    (omx_interfaces__srv__GenerateTrajectory_Request__Sequence *)(untyped_member);
  omx_interfaces__srv__GenerateTrajectory_Request__Sequence__fini(member);
  return omx_interfaces__srv__GenerateTrajectory_Request__Sequence__init(member, size);
}

size_t omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__size_function__GenerateTrajectory_Event__response(
  const void * untyped_member)
{
  const omx_interfaces__srv__GenerateTrajectory_Response__Sequence * member =
    (const omx_interfaces__srv__GenerateTrajectory_Response__Sequence *)(untyped_member);
  return member->size;
}

const void * omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__get_const_function__GenerateTrajectory_Event__response(
  const void * untyped_member, size_t index)
{
  const omx_interfaces__srv__GenerateTrajectory_Response__Sequence * member =
    (const omx_interfaces__srv__GenerateTrajectory_Response__Sequence *)(untyped_member);
  return &member->data[index];
}

void * omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__get_function__GenerateTrajectory_Event__response(
  void * untyped_member, size_t index)
{
  omx_interfaces__srv__GenerateTrajectory_Response__Sequence * member =
    (omx_interfaces__srv__GenerateTrajectory_Response__Sequence *)(untyped_member);
  return &member->data[index];
}

void omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__fetch_function__GenerateTrajectory_Event__response(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const omx_interfaces__srv__GenerateTrajectory_Response * item =
    ((const omx_interfaces__srv__GenerateTrajectory_Response *)
    omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__get_const_function__GenerateTrajectory_Event__response(untyped_member, index));
  omx_interfaces__srv__GenerateTrajectory_Response * value =
    (omx_interfaces__srv__GenerateTrajectory_Response *)(untyped_value);
  *value = *item;
}

void omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__assign_function__GenerateTrajectory_Event__response(
  void * untyped_member, size_t index, const void * untyped_value)
{
  omx_interfaces__srv__GenerateTrajectory_Response * item =
    ((omx_interfaces__srv__GenerateTrajectory_Response *)
    omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__get_function__GenerateTrajectory_Event__response(untyped_member, index));
  const omx_interfaces__srv__GenerateTrajectory_Response * value =
    (const omx_interfaces__srv__GenerateTrajectory_Response *)(untyped_value);
  *item = *value;
}

bool omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__resize_function__GenerateTrajectory_Event__response(
  void * untyped_member, size_t size)
{
  omx_interfaces__srv__GenerateTrajectory_Response__Sequence * member =
    (omx_interfaces__srv__GenerateTrajectory_Response__Sequence *)(untyped_member);
  omx_interfaces__srv__GenerateTrajectory_Response__Sequence__fini(member);
  return omx_interfaces__srv__GenerateTrajectory_Response__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__GenerateTrajectory_Event_message_member_array[3] = {
  {
    "info",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(omx_interfaces__srv__GenerateTrajectory_Event, info),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "request",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    true,  // is array
    1,  // array size
    true,  // is upper bound
    offsetof(omx_interfaces__srv__GenerateTrajectory_Event, request),  // bytes offset in struct
    NULL,  // default value
    omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__size_function__GenerateTrajectory_Event__request,  // size() function pointer
    omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__get_const_function__GenerateTrajectory_Event__request,  // get_const(index) function pointer
    omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__get_function__GenerateTrajectory_Event__request,  // get(index) function pointer
    omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__fetch_function__GenerateTrajectory_Event__request,  // fetch(index, &value) function pointer
    omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__assign_function__GenerateTrajectory_Event__request,  // assign(index, value) function pointer
    omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__resize_function__GenerateTrajectory_Event__request  // resize(index) function pointer
  },
  {
    "response",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    true,  // is array
    1,  // array size
    true,  // is upper bound
    offsetof(omx_interfaces__srv__GenerateTrajectory_Event, response),  // bytes offset in struct
    NULL,  // default value
    omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__size_function__GenerateTrajectory_Event__response,  // size() function pointer
    omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__get_const_function__GenerateTrajectory_Event__response,  // get_const(index) function pointer
    omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__get_function__GenerateTrajectory_Event__response,  // get(index) function pointer
    omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__fetch_function__GenerateTrajectory_Event__response,  // fetch(index, &value) function pointer
    omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__assign_function__GenerateTrajectory_Event__response,  // assign(index, value) function pointer
    omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__resize_function__GenerateTrajectory_Event__response  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__GenerateTrajectory_Event_message_members = {
  "omx_interfaces__srv",  // message namespace
  "GenerateTrajectory_Event",  // message name
  3,  // number of fields
  sizeof(omx_interfaces__srv__GenerateTrajectory_Event),
  false,  // has_any_key_member_
  omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__GenerateTrajectory_Event_message_member_array,  // message members
  omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__GenerateTrajectory_Event_init_function,  // function to initialize message memory (memory has to be allocated)
  omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__GenerateTrajectory_Event_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__GenerateTrajectory_Event_message_type_support_handle = {
  0,
  &omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__GenerateTrajectory_Event_message_members,
  get_message_typesupport_handle_function,
  &omx_interfaces__srv__GenerateTrajectory_Event__get_type_hash,
  &omx_interfaces__srv__GenerateTrajectory_Event__get_type_description,
  &omx_interfaces__srv__GenerateTrajectory_Event__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_omx_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, omx_interfaces, srv, GenerateTrajectory_Event)() {
  omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__GenerateTrajectory_Event_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, service_msgs, msg, ServiceEventInfo)();
  omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__GenerateTrajectory_Event_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, omx_interfaces, srv, GenerateTrajectory_Request)();
  omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__GenerateTrajectory_Event_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, omx_interfaces, srv, GenerateTrajectory_Response)();
  if (!omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__GenerateTrajectory_Event_message_type_support_handle.typesupport_identifier) {
    omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__GenerateTrajectory_Event_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__GenerateTrajectory_Event_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "omx_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "omx_interfaces/srv/detail/generate_trajectory__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers omx_interfaces__srv__detail__generate_trajectory__rosidl_typesupport_introspection_c__GenerateTrajectory_service_members = {
  "omx_interfaces__srv",  // service namespace
  "GenerateTrajectory",  // service name
  // the following fields are initialized below on first access
  NULL,  // request message
  // omx_interfaces__srv__detail__generate_trajectory__rosidl_typesupport_introspection_c__GenerateTrajectory_Request_message_type_support_handle,
  NULL,  // response message
  // omx_interfaces__srv__detail__generate_trajectory__rosidl_typesupport_introspection_c__GenerateTrajectory_Response_message_type_support_handle
  NULL  // event_message
  // omx_interfaces__srv__detail__generate_trajectory__rosidl_typesupport_introspection_c__GenerateTrajectory_Response_message_type_support_handle
};


static rosidl_service_type_support_t omx_interfaces__srv__detail__generate_trajectory__rosidl_typesupport_introspection_c__GenerateTrajectory_service_type_support_handle = {
  0,
  &omx_interfaces__srv__detail__generate_trajectory__rosidl_typesupport_introspection_c__GenerateTrajectory_service_members,
  get_service_typesupport_handle_function,
  &omx_interfaces__srv__GenerateTrajectory_Request__rosidl_typesupport_introspection_c__GenerateTrajectory_Request_message_type_support_handle,
  &omx_interfaces__srv__GenerateTrajectory_Response__rosidl_typesupport_introspection_c__GenerateTrajectory_Response_message_type_support_handle,
  &omx_interfaces__srv__GenerateTrajectory_Event__rosidl_typesupport_introspection_c__GenerateTrajectory_Event_message_type_support_handle,
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_CREATE_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    omx_interfaces,
    srv,
    GenerateTrajectory
  ),
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_DESTROY_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    omx_interfaces,
    srv,
    GenerateTrajectory
  ),
  &omx_interfaces__srv__GenerateTrajectory__get_type_hash,
  &omx_interfaces__srv__GenerateTrajectory__get_type_description,
  &omx_interfaces__srv__GenerateTrajectory__get_type_description_sources,
};

// Forward declaration of message type support functions for service members
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, omx_interfaces, srv, GenerateTrajectory_Request)(void);

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, omx_interfaces, srv, GenerateTrajectory_Response)(void);

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, omx_interfaces, srv, GenerateTrajectory_Event)(void);

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_omx_interfaces
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, omx_interfaces, srv, GenerateTrajectory)(void) {
  if (!omx_interfaces__srv__detail__generate_trajectory__rosidl_typesupport_introspection_c__GenerateTrajectory_service_type_support_handle.typesupport_identifier) {
    omx_interfaces__srv__detail__generate_trajectory__rosidl_typesupport_introspection_c__GenerateTrajectory_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)omx_interfaces__srv__detail__generate_trajectory__rosidl_typesupport_introspection_c__GenerateTrajectory_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, omx_interfaces, srv, GenerateTrajectory_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, omx_interfaces, srv, GenerateTrajectory_Response)()->data;
  }
  if (!service_members->event_members_) {
    service_members->event_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, omx_interfaces, srv, GenerateTrajectory_Event)()->data;
  }

  return &omx_interfaces__srv__detail__generate_trajectory__rosidl_typesupport_introspection_c__GenerateTrajectory_service_type_support_handle;
}
