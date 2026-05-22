// generated from rosidl_typesupport_c/resource/idl__type_support.cpp.em
// with input from omx_interfaces:srv/GenerateTrajectory.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "omx_interfaces/srv/detail/generate_trajectory__struct.h"
#include "omx_interfaces/srv/detail/generate_trajectory__type_support.h"
#include "omx_interfaces/srv/detail/generate_trajectory__functions.h"
#include "rosidl_typesupport_c/identifier.h"
#include "rosidl_typesupport_c/message_type_support_dispatch.h"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_c/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace omx_interfaces
{

namespace srv
{

namespace rosidl_typesupport_c
{

typedef struct _GenerateTrajectory_Request_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _GenerateTrajectory_Request_type_support_ids_t;

static const _GenerateTrajectory_Request_type_support_ids_t _GenerateTrajectory_Request_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _GenerateTrajectory_Request_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _GenerateTrajectory_Request_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _GenerateTrajectory_Request_type_support_symbol_names_t _GenerateTrajectory_Request_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, omx_interfaces, srv, GenerateTrajectory_Request)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, omx_interfaces, srv, GenerateTrajectory_Request)),
  }
};

typedef struct _GenerateTrajectory_Request_type_support_data_t
{
  void * data[2];
} _GenerateTrajectory_Request_type_support_data_t;

static _GenerateTrajectory_Request_type_support_data_t _GenerateTrajectory_Request_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _GenerateTrajectory_Request_message_typesupport_map = {
  2,
  "omx_interfaces",
  &_GenerateTrajectory_Request_message_typesupport_ids.typesupport_identifier[0],
  &_GenerateTrajectory_Request_message_typesupport_symbol_names.symbol_name[0],
  &_GenerateTrajectory_Request_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t GenerateTrajectory_Request_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_GenerateTrajectory_Request_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &omx_interfaces__srv__GenerateTrajectory_Request__get_type_hash,
  &omx_interfaces__srv__GenerateTrajectory_Request__get_type_description,
  &omx_interfaces__srv__GenerateTrajectory_Request__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace omx_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, omx_interfaces, srv, GenerateTrajectory_Request)() {
  return &::omx_interfaces::srv::rosidl_typesupport_c::GenerateTrajectory_Request_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "omx_interfaces/srv/detail/generate_trajectory__struct.h"
// already included above
// #include "omx_interfaces/srv/detail/generate_trajectory__type_support.h"
// already included above
// #include "omx_interfaces/srv/detail/generate_trajectory__functions.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace omx_interfaces
{

namespace srv
{

namespace rosidl_typesupport_c
{

typedef struct _GenerateTrajectory_Response_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _GenerateTrajectory_Response_type_support_ids_t;

static const _GenerateTrajectory_Response_type_support_ids_t _GenerateTrajectory_Response_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _GenerateTrajectory_Response_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _GenerateTrajectory_Response_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _GenerateTrajectory_Response_type_support_symbol_names_t _GenerateTrajectory_Response_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, omx_interfaces, srv, GenerateTrajectory_Response)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, omx_interfaces, srv, GenerateTrajectory_Response)),
  }
};

typedef struct _GenerateTrajectory_Response_type_support_data_t
{
  void * data[2];
} _GenerateTrajectory_Response_type_support_data_t;

static _GenerateTrajectory_Response_type_support_data_t _GenerateTrajectory_Response_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _GenerateTrajectory_Response_message_typesupport_map = {
  2,
  "omx_interfaces",
  &_GenerateTrajectory_Response_message_typesupport_ids.typesupport_identifier[0],
  &_GenerateTrajectory_Response_message_typesupport_symbol_names.symbol_name[0],
  &_GenerateTrajectory_Response_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t GenerateTrajectory_Response_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_GenerateTrajectory_Response_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &omx_interfaces__srv__GenerateTrajectory_Response__get_type_hash,
  &omx_interfaces__srv__GenerateTrajectory_Response__get_type_description,
  &omx_interfaces__srv__GenerateTrajectory_Response__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace omx_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, omx_interfaces, srv, GenerateTrajectory_Response)() {
  return &::omx_interfaces::srv::rosidl_typesupport_c::GenerateTrajectory_Response_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "omx_interfaces/srv/detail/generate_trajectory__struct.h"
// already included above
// #include "omx_interfaces/srv/detail/generate_trajectory__type_support.h"
// already included above
// #include "omx_interfaces/srv/detail/generate_trajectory__functions.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
// already included above
// #include "rosidl_typesupport_c/message_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_c/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace omx_interfaces
{

namespace srv
{

namespace rosidl_typesupport_c
{

typedef struct _GenerateTrajectory_Event_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _GenerateTrajectory_Event_type_support_ids_t;

static const _GenerateTrajectory_Event_type_support_ids_t _GenerateTrajectory_Event_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _GenerateTrajectory_Event_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _GenerateTrajectory_Event_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _GenerateTrajectory_Event_type_support_symbol_names_t _GenerateTrajectory_Event_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, omx_interfaces, srv, GenerateTrajectory_Event)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, omx_interfaces, srv, GenerateTrajectory_Event)),
  }
};

typedef struct _GenerateTrajectory_Event_type_support_data_t
{
  void * data[2];
} _GenerateTrajectory_Event_type_support_data_t;

static _GenerateTrajectory_Event_type_support_data_t _GenerateTrajectory_Event_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _GenerateTrajectory_Event_message_typesupport_map = {
  2,
  "omx_interfaces",
  &_GenerateTrajectory_Event_message_typesupport_ids.typesupport_identifier[0],
  &_GenerateTrajectory_Event_message_typesupport_symbol_names.symbol_name[0],
  &_GenerateTrajectory_Event_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t GenerateTrajectory_Event_message_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_GenerateTrajectory_Event_message_typesupport_map),
  rosidl_typesupport_c__get_message_typesupport_handle_function,
  &omx_interfaces__srv__GenerateTrajectory_Event__get_type_hash,
  &omx_interfaces__srv__GenerateTrajectory_Event__get_type_description,
  &omx_interfaces__srv__GenerateTrajectory_Event__get_type_description_sources,
};

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace omx_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_c, omx_interfaces, srv, GenerateTrajectory_Event)() {
  return &::omx_interfaces::srv::rosidl_typesupport_c::GenerateTrajectory_Event_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "cstddef"
#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "omx_interfaces/srv/detail/generate_trajectory__type_support.h"
// already included above
// #include "rosidl_typesupport_c/identifier.h"
#include "rosidl_typesupport_c/service_type_support_dispatch.h"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
#include "service_msgs/msg/service_event_info.h"
#include "builtin_interfaces/msg/time.h"

namespace omx_interfaces
{

namespace srv
{

namespace rosidl_typesupport_c
{
typedef struct _GenerateTrajectory_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _GenerateTrajectory_type_support_ids_t;

static const _GenerateTrajectory_type_support_ids_t _GenerateTrajectory_service_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_c",  // ::rosidl_typesupport_fastrtps_c::typesupport_identifier,
    "rosidl_typesupport_introspection_c",  // ::rosidl_typesupport_introspection_c::typesupport_identifier,
  }
};

typedef struct _GenerateTrajectory_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _GenerateTrajectory_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _GenerateTrajectory_type_support_symbol_names_t _GenerateTrajectory_service_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, omx_interfaces, srv, GenerateTrajectory)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, omx_interfaces, srv, GenerateTrajectory)),
  }
};

typedef struct _GenerateTrajectory_type_support_data_t
{
  void * data[2];
} _GenerateTrajectory_type_support_data_t;

static _GenerateTrajectory_type_support_data_t _GenerateTrajectory_service_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _GenerateTrajectory_service_typesupport_map = {
  2,
  "omx_interfaces",
  &_GenerateTrajectory_service_typesupport_ids.typesupport_identifier[0],
  &_GenerateTrajectory_service_typesupport_symbol_names.symbol_name[0],
  &_GenerateTrajectory_service_typesupport_data.data[0],
};

static const rosidl_service_type_support_t GenerateTrajectory_service_type_support_handle = {
  rosidl_typesupport_c__typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_GenerateTrajectory_service_typesupport_map),
  rosidl_typesupport_c__get_service_typesupport_handle_function,
  &GenerateTrajectory_Request_message_type_support_handle,
  &GenerateTrajectory_Response_message_type_support_handle,
  &GenerateTrajectory_Event_message_type_support_handle,
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

}  // namespace rosidl_typesupport_c

}  // namespace srv

}  // namespace omx_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_c, omx_interfaces, srv, GenerateTrajectory)() {
  return &::omx_interfaces::srv::rosidl_typesupport_c::GenerateTrajectory_service_type_support_handle;
}

#ifdef __cplusplus
}
#endif
