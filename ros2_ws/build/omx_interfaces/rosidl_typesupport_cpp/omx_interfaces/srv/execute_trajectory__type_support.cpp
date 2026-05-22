// generated from rosidl_typesupport_cpp/resource/idl__type_support.cpp.em
// with input from omx_interfaces:srv/ExecuteTrajectory.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "omx_interfaces/srv/detail/execute_trajectory__functions.h"
#include "omx_interfaces/srv/detail/execute_trajectory__struct.hpp"
#include "rosidl_typesupport_cpp/identifier.hpp"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_cpp/message_type_support_dispatch.hpp"
#include "rosidl_typesupport_cpp/visibility_control.h"
#include "rosidl_typesupport_interface/macros.h"

namespace omx_interfaces
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _ExecuteTrajectory_Request_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _ExecuteTrajectory_Request_type_support_ids_t;

static const _ExecuteTrajectory_Request_type_support_ids_t _ExecuteTrajectory_Request_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _ExecuteTrajectory_Request_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _ExecuteTrajectory_Request_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _ExecuteTrajectory_Request_type_support_symbol_names_t _ExecuteTrajectory_Request_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, omx_interfaces, srv, ExecuteTrajectory_Request)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, omx_interfaces, srv, ExecuteTrajectory_Request)),
  }
};

typedef struct _ExecuteTrajectory_Request_type_support_data_t
{
  void * data[2];
} _ExecuteTrajectory_Request_type_support_data_t;

static _ExecuteTrajectory_Request_type_support_data_t _ExecuteTrajectory_Request_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _ExecuteTrajectory_Request_message_typesupport_map = {
  2,
  "omx_interfaces",
  &_ExecuteTrajectory_Request_message_typesupport_ids.typesupport_identifier[0],
  &_ExecuteTrajectory_Request_message_typesupport_symbol_names.symbol_name[0],
  &_ExecuteTrajectory_Request_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t ExecuteTrajectory_Request_message_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_ExecuteTrajectory_Request_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
  &omx_interfaces__srv__ExecuteTrajectory_Request__get_type_hash,
  &omx_interfaces__srv__ExecuteTrajectory_Request__get_type_description,
  &omx_interfaces__srv__ExecuteTrajectory_Request__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace omx_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<omx_interfaces::srv::ExecuteTrajectory_Request>()
{
  return &::omx_interfaces::srv::rosidl_typesupport_cpp::ExecuteTrajectory_Request_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, omx_interfaces, srv, ExecuteTrajectory_Request)() {
  return get_message_type_support_handle<omx_interfaces::srv::ExecuteTrajectory_Request>();
}

#ifdef __cplusplus
}
#endif
}  // namespace rosidl_typesupport_cpp

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "omx_interfaces/srv/detail/execute_trajectory__functions.h"
// already included above
// #include "omx_interfaces/srv/detail/execute_trajectory__struct.hpp"
// already included above
// #include "rosidl_typesupport_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support_dispatch.hpp"
// already included above
// #include "rosidl_typesupport_cpp/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace omx_interfaces
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _ExecuteTrajectory_Response_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _ExecuteTrajectory_Response_type_support_ids_t;

static const _ExecuteTrajectory_Response_type_support_ids_t _ExecuteTrajectory_Response_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _ExecuteTrajectory_Response_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _ExecuteTrajectory_Response_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _ExecuteTrajectory_Response_type_support_symbol_names_t _ExecuteTrajectory_Response_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, omx_interfaces, srv, ExecuteTrajectory_Response)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, omx_interfaces, srv, ExecuteTrajectory_Response)),
  }
};

typedef struct _ExecuteTrajectory_Response_type_support_data_t
{
  void * data[2];
} _ExecuteTrajectory_Response_type_support_data_t;

static _ExecuteTrajectory_Response_type_support_data_t _ExecuteTrajectory_Response_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _ExecuteTrajectory_Response_message_typesupport_map = {
  2,
  "omx_interfaces",
  &_ExecuteTrajectory_Response_message_typesupport_ids.typesupport_identifier[0],
  &_ExecuteTrajectory_Response_message_typesupport_symbol_names.symbol_name[0],
  &_ExecuteTrajectory_Response_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t ExecuteTrajectory_Response_message_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_ExecuteTrajectory_Response_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
  &omx_interfaces__srv__ExecuteTrajectory_Response__get_type_hash,
  &omx_interfaces__srv__ExecuteTrajectory_Response__get_type_description,
  &omx_interfaces__srv__ExecuteTrajectory_Response__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace omx_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<omx_interfaces::srv::ExecuteTrajectory_Response>()
{
  return &::omx_interfaces::srv::rosidl_typesupport_cpp::ExecuteTrajectory_Response_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, omx_interfaces, srv, ExecuteTrajectory_Response)() {
  return get_message_type_support_handle<omx_interfaces::srv::ExecuteTrajectory_Response>();
}

#ifdef __cplusplus
}
#endif
}  // namespace rosidl_typesupport_cpp

// already included above
// #include "cstddef"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "omx_interfaces/srv/detail/execute_trajectory__functions.h"
// already included above
// #include "omx_interfaces/srv/detail/execute_trajectory__struct.hpp"
// already included above
// #include "rosidl_typesupport_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support_dispatch.hpp"
// already included above
// #include "rosidl_typesupport_cpp/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace omx_interfaces
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _ExecuteTrajectory_Event_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _ExecuteTrajectory_Event_type_support_ids_t;

static const _ExecuteTrajectory_Event_type_support_ids_t _ExecuteTrajectory_Event_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _ExecuteTrajectory_Event_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _ExecuteTrajectory_Event_type_support_symbol_names_t;

#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _ExecuteTrajectory_Event_type_support_symbol_names_t _ExecuteTrajectory_Event_message_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, omx_interfaces, srv, ExecuteTrajectory_Event)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, omx_interfaces, srv, ExecuteTrajectory_Event)),
  }
};

typedef struct _ExecuteTrajectory_Event_type_support_data_t
{
  void * data[2];
} _ExecuteTrajectory_Event_type_support_data_t;

static _ExecuteTrajectory_Event_type_support_data_t _ExecuteTrajectory_Event_message_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _ExecuteTrajectory_Event_message_typesupport_map = {
  2,
  "omx_interfaces",
  &_ExecuteTrajectory_Event_message_typesupport_ids.typesupport_identifier[0],
  &_ExecuteTrajectory_Event_message_typesupport_symbol_names.symbol_name[0],
  &_ExecuteTrajectory_Event_message_typesupport_data.data[0],
};

static const rosidl_message_type_support_t ExecuteTrajectory_Event_message_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_ExecuteTrajectory_Event_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
  &omx_interfaces__srv__ExecuteTrajectory_Event__get_type_hash,
  &omx_interfaces__srv__ExecuteTrajectory_Event__get_type_description,
  &omx_interfaces__srv__ExecuteTrajectory_Event__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace omx_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<omx_interfaces::srv::ExecuteTrajectory_Event>()
{
  return &::omx_interfaces::srv::rosidl_typesupport_cpp::ExecuteTrajectory_Event_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, omx_interfaces, srv, ExecuteTrajectory_Event)() {
  return get_message_type_support_handle<omx_interfaces::srv::ExecuteTrajectory_Event>();
}

#ifdef __cplusplus
}
#endif
}  // namespace rosidl_typesupport_cpp

// already included above
// #include "cstddef"
#include "rosidl_runtime_c/service_type_support_struct.h"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "omx_interfaces/srv/detail/execute_trajectory__struct.hpp"
// already included above
// #include "rosidl_typesupport_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_c/type_support_map.h"
#include "rosidl_typesupport_cpp/service_type_support_dispatch.hpp"
// already included above
// #include "rosidl_typesupport_cpp/visibility_control.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"

namespace omx_interfaces
{

namespace srv
{

namespace rosidl_typesupport_cpp
{

typedef struct _ExecuteTrajectory_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _ExecuteTrajectory_type_support_ids_t;

static const _ExecuteTrajectory_type_support_ids_t _ExecuteTrajectory_service_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  }
};

typedef struct _ExecuteTrajectory_type_support_symbol_names_t
{
  const char * symbol_name[2];
} _ExecuteTrajectory_type_support_symbol_names_t;
#define STRINGIFY_(s) #s
#define STRINGIFY(s) STRINGIFY_(s)

static const _ExecuteTrajectory_type_support_symbol_names_t _ExecuteTrajectory_service_typesupport_symbol_names = {
  {
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, omx_interfaces, srv, ExecuteTrajectory)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, omx_interfaces, srv, ExecuteTrajectory)),
  }
};

typedef struct _ExecuteTrajectory_type_support_data_t
{
  void * data[2];
} _ExecuteTrajectory_type_support_data_t;

static _ExecuteTrajectory_type_support_data_t _ExecuteTrajectory_service_typesupport_data = {
  {
    0,  // will store the shared library later
    0,  // will store the shared library later
  }
};

static const type_support_map_t _ExecuteTrajectory_service_typesupport_map = {
  2,
  "omx_interfaces",
  &_ExecuteTrajectory_service_typesupport_ids.typesupport_identifier[0],
  &_ExecuteTrajectory_service_typesupport_symbol_names.symbol_name[0],
  &_ExecuteTrajectory_service_typesupport_data.data[0],
};

static const rosidl_service_type_support_t ExecuteTrajectory_service_type_support_handle = {
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_ExecuteTrajectory_service_typesupport_map),
  ::rosidl_typesupport_cpp::get_service_typesupport_handle_function,
  ::rosidl_typesupport_cpp::get_message_type_support_handle<omx_interfaces::srv::ExecuteTrajectory_Request>(),
  ::rosidl_typesupport_cpp::get_message_type_support_handle<omx_interfaces::srv::ExecuteTrajectory_Response>(),
  ::rosidl_typesupport_cpp::get_message_type_support_handle<omx_interfaces::srv::ExecuteTrajectory_Event>(),
  &::rosidl_typesupport_cpp::service_create_event_message<omx_interfaces::srv::ExecuteTrajectory>,
  &::rosidl_typesupport_cpp::service_destroy_event_message<omx_interfaces::srv::ExecuteTrajectory>,
  &omx_interfaces__srv__ExecuteTrajectory__get_type_hash,
  &omx_interfaces__srv__ExecuteTrajectory__get_type_description,
  &omx_interfaces__srv__ExecuteTrajectory__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace omx_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_service_type_support_t *
get_service_type_support_handle<omx_interfaces::srv::ExecuteTrajectory>()
{
  return &::omx_interfaces::srv::rosidl_typesupport_cpp::ExecuteTrajectory_service_type_support_handle;
}

}  // namespace rosidl_typesupport_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_cpp, omx_interfaces, srv, ExecuteTrajectory)() {
  return ::rosidl_typesupport_cpp::get_service_type_support_handle<omx_interfaces::srv::ExecuteTrajectory>();
}

#ifdef __cplusplus
}
#endif
