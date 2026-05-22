// generated from rosidl_typesupport_cpp/resource/idl__type_support.cpp.em
// with input from omx_interfaces:srv/GenerateTrajectory.idl
// generated code does not contain a copyright notice

#include "cstddef"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "omx_interfaces/srv/detail/generate_trajectory__functions.h"
#include "omx_interfaces/srv/detail/generate_trajectory__struct.hpp"
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

typedef struct _GenerateTrajectory_Request_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _GenerateTrajectory_Request_type_support_ids_t;

static const _GenerateTrajectory_Request_type_support_ids_t _GenerateTrajectory_Request_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
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
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, omx_interfaces, srv, GenerateTrajectory_Request)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, omx_interfaces, srv, GenerateTrajectory_Request)),
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
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_GenerateTrajectory_Request_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
  &omx_interfaces__srv__GenerateTrajectory_Request__get_type_hash,
  &omx_interfaces__srv__GenerateTrajectory_Request__get_type_description,
  &omx_interfaces__srv__GenerateTrajectory_Request__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace omx_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<omx_interfaces::srv::GenerateTrajectory_Request>()
{
  return &::omx_interfaces::srv::rosidl_typesupport_cpp::GenerateTrajectory_Request_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, omx_interfaces, srv, GenerateTrajectory_Request)() {
  return get_message_type_support_handle<omx_interfaces::srv::GenerateTrajectory_Request>();
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
// #include "omx_interfaces/srv/detail/generate_trajectory__functions.h"
// already included above
// #include "omx_interfaces/srv/detail/generate_trajectory__struct.hpp"
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

typedef struct _GenerateTrajectory_Response_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _GenerateTrajectory_Response_type_support_ids_t;

static const _GenerateTrajectory_Response_type_support_ids_t _GenerateTrajectory_Response_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
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
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, omx_interfaces, srv, GenerateTrajectory_Response)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, omx_interfaces, srv, GenerateTrajectory_Response)),
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
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_GenerateTrajectory_Response_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
  &omx_interfaces__srv__GenerateTrajectory_Response__get_type_hash,
  &omx_interfaces__srv__GenerateTrajectory_Response__get_type_description,
  &omx_interfaces__srv__GenerateTrajectory_Response__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace omx_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<omx_interfaces::srv::GenerateTrajectory_Response>()
{
  return &::omx_interfaces::srv::rosidl_typesupport_cpp::GenerateTrajectory_Response_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, omx_interfaces, srv, GenerateTrajectory_Response)() {
  return get_message_type_support_handle<omx_interfaces::srv::GenerateTrajectory_Response>();
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
// #include "omx_interfaces/srv/detail/generate_trajectory__functions.h"
// already included above
// #include "omx_interfaces/srv/detail/generate_trajectory__struct.hpp"
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

typedef struct _GenerateTrajectory_Event_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _GenerateTrajectory_Event_type_support_ids_t;

static const _GenerateTrajectory_Event_type_support_ids_t _GenerateTrajectory_Event_message_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
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
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, omx_interfaces, srv, GenerateTrajectory_Event)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, omx_interfaces, srv, GenerateTrajectory_Event)),
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
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_GenerateTrajectory_Event_message_typesupport_map),
  ::rosidl_typesupport_cpp::get_message_typesupport_handle_function,
  &omx_interfaces__srv__GenerateTrajectory_Event__get_type_hash,
  &omx_interfaces__srv__GenerateTrajectory_Event__get_type_description,
  &omx_interfaces__srv__GenerateTrajectory_Event__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace omx_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<omx_interfaces::srv::GenerateTrajectory_Event>()
{
  return &::omx_interfaces::srv::rosidl_typesupport_cpp::GenerateTrajectory_Event_message_type_support_handle;
}

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_cpp, omx_interfaces, srv, GenerateTrajectory_Event)() {
  return get_message_type_support_handle<omx_interfaces::srv::GenerateTrajectory_Event>();
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
// #include "omx_interfaces/srv/detail/generate_trajectory__struct.hpp"
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

typedef struct _GenerateTrajectory_type_support_ids_t
{
  const char * typesupport_identifier[2];
} _GenerateTrajectory_type_support_ids_t;

static const _GenerateTrajectory_type_support_ids_t _GenerateTrajectory_service_typesupport_ids = {
  {
    "rosidl_typesupport_fastrtps_cpp",  // ::rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
    "rosidl_typesupport_introspection_cpp",  // ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
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
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, omx_interfaces, srv, GenerateTrajectory)),
    STRINGIFY(ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, omx_interfaces, srv, GenerateTrajectory)),
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
  ::rosidl_typesupport_cpp::typesupport_identifier,
  reinterpret_cast<const type_support_map_t *>(&_GenerateTrajectory_service_typesupport_map),
  ::rosidl_typesupport_cpp::get_service_typesupport_handle_function,
  ::rosidl_typesupport_cpp::get_message_type_support_handle<omx_interfaces::srv::GenerateTrajectory_Request>(),
  ::rosidl_typesupport_cpp::get_message_type_support_handle<omx_interfaces::srv::GenerateTrajectory_Response>(),
  ::rosidl_typesupport_cpp::get_message_type_support_handle<omx_interfaces::srv::GenerateTrajectory_Event>(),
  &::rosidl_typesupport_cpp::service_create_event_message<omx_interfaces::srv::GenerateTrajectory>,
  &::rosidl_typesupport_cpp::service_destroy_event_message<omx_interfaces::srv::GenerateTrajectory>,
  &omx_interfaces__srv__GenerateTrajectory__get_type_hash,
  &omx_interfaces__srv__GenerateTrajectory__get_type_description,
  &omx_interfaces__srv__GenerateTrajectory__get_type_description_sources,
};

}  // namespace rosidl_typesupport_cpp

}  // namespace srv

}  // namespace omx_interfaces

namespace rosidl_typesupport_cpp
{

template<>
ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_service_type_support_t *
get_service_type_support_handle<omx_interfaces::srv::GenerateTrajectory>()
{
  return &::omx_interfaces::srv::rosidl_typesupport_cpp::GenerateTrajectory_service_type_support_handle;
}

}  // namespace rosidl_typesupport_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_CPP_PUBLIC
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_cpp, omx_interfaces, srv, GenerateTrajectory)() {
  return ::rosidl_typesupport_cpp::get_service_type_support_handle<omx_interfaces::srv::GenerateTrajectory>();
}

#ifdef __cplusplus
}
#endif
