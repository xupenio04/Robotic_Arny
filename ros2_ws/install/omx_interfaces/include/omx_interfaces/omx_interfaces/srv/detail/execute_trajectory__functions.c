// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from omx_interfaces:srv/ExecuteTrajectory.idl
// generated code does not contain a copyright notice
#include "omx_interfaces/srv/detail/execute_trajectory__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `trajectory`
#include "trajectory_msgs/msg/detail/joint_trajectory__functions.h"

bool
omx_interfaces__srv__ExecuteTrajectory_Request__init(omx_interfaces__srv__ExecuteTrajectory_Request * msg)
{
  if (!msg) {
    return false;
  }
  // trajectory
  if (!trajectory_msgs__msg__JointTrajectory__init(&msg->trajectory)) {
    omx_interfaces__srv__ExecuteTrajectory_Request__fini(msg);
    return false;
  }
  return true;
}

void
omx_interfaces__srv__ExecuteTrajectory_Request__fini(omx_interfaces__srv__ExecuteTrajectory_Request * msg)
{
  if (!msg) {
    return;
  }
  // trajectory
  trajectory_msgs__msg__JointTrajectory__fini(&msg->trajectory);
}

bool
omx_interfaces__srv__ExecuteTrajectory_Request__are_equal(const omx_interfaces__srv__ExecuteTrajectory_Request * lhs, const omx_interfaces__srv__ExecuteTrajectory_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // trajectory
  if (!trajectory_msgs__msg__JointTrajectory__are_equal(
      &(lhs->trajectory), &(rhs->trajectory)))
  {
    return false;
  }
  return true;
}

bool
omx_interfaces__srv__ExecuteTrajectory_Request__copy(
  const omx_interfaces__srv__ExecuteTrajectory_Request * input,
  omx_interfaces__srv__ExecuteTrajectory_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // trajectory
  if (!trajectory_msgs__msg__JointTrajectory__copy(
      &(input->trajectory), &(output->trajectory)))
  {
    return false;
  }
  return true;
}

omx_interfaces__srv__ExecuteTrajectory_Request *
omx_interfaces__srv__ExecuteTrajectory_Request__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  omx_interfaces__srv__ExecuteTrajectory_Request * msg = (omx_interfaces__srv__ExecuteTrajectory_Request *)allocator.allocate(sizeof(omx_interfaces__srv__ExecuteTrajectory_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(omx_interfaces__srv__ExecuteTrajectory_Request));
  bool success = omx_interfaces__srv__ExecuteTrajectory_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
omx_interfaces__srv__ExecuteTrajectory_Request__destroy(omx_interfaces__srv__ExecuteTrajectory_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    omx_interfaces__srv__ExecuteTrajectory_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__init(omx_interfaces__srv__ExecuteTrajectory_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  omx_interfaces__srv__ExecuteTrajectory_Request * data = NULL;

  if (size) {
    data = (omx_interfaces__srv__ExecuteTrajectory_Request *)allocator.zero_allocate(size, sizeof(omx_interfaces__srv__ExecuteTrajectory_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = omx_interfaces__srv__ExecuteTrajectory_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        omx_interfaces__srv__ExecuteTrajectory_Request__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__fini(omx_interfaces__srv__ExecuteTrajectory_Request__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      omx_interfaces__srv__ExecuteTrajectory_Request__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

omx_interfaces__srv__ExecuteTrajectory_Request__Sequence *
omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  omx_interfaces__srv__ExecuteTrajectory_Request__Sequence * array = (omx_interfaces__srv__ExecuteTrajectory_Request__Sequence *)allocator.allocate(sizeof(omx_interfaces__srv__ExecuteTrajectory_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__destroy(omx_interfaces__srv__ExecuteTrajectory_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__are_equal(const omx_interfaces__srv__ExecuteTrajectory_Request__Sequence * lhs, const omx_interfaces__srv__ExecuteTrajectory_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!omx_interfaces__srv__ExecuteTrajectory_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__copy(
  const omx_interfaces__srv__ExecuteTrajectory_Request__Sequence * input,
  omx_interfaces__srv__ExecuteTrajectory_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(omx_interfaces__srv__ExecuteTrajectory_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    omx_interfaces__srv__ExecuteTrajectory_Request * data =
      (omx_interfaces__srv__ExecuteTrajectory_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!omx_interfaces__srv__ExecuteTrajectory_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          omx_interfaces__srv__ExecuteTrajectory_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!omx_interfaces__srv__ExecuteTrajectory_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `message`
#include "rosidl_runtime_c/string_functions.h"

bool
omx_interfaces__srv__ExecuteTrajectory_Response__init(omx_interfaces__srv__ExecuteTrajectory_Response * msg)
{
  if (!msg) {
    return false;
  }
  // success
  // message
  if (!rosidl_runtime_c__String__init(&msg->message)) {
    omx_interfaces__srv__ExecuteTrajectory_Response__fini(msg);
    return false;
  }
  return true;
}

void
omx_interfaces__srv__ExecuteTrajectory_Response__fini(omx_interfaces__srv__ExecuteTrajectory_Response * msg)
{
  if (!msg) {
    return;
  }
  // success
  // message
  rosidl_runtime_c__String__fini(&msg->message);
}

bool
omx_interfaces__srv__ExecuteTrajectory_Response__are_equal(const omx_interfaces__srv__ExecuteTrajectory_Response * lhs, const omx_interfaces__srv__ExecuteTrajectory_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  // message
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->message), &(rhs->message)))
  {
    return false;
  }
  return true;
}

bool
omx_interfaces__srv__ExecuteTrajectory_Response__copy(
  const omx_interfaces__srv__ExecuteTrajectory_Response * input,
  omx_interfaces__srv__ExecuteTrajectory_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // success
  output->success = input->success;
  // message
  if (!rosidl_runtime_c__String__copy(
      &(input->message), &(output->message)))
  {
    return false;
  }
  return true;
}

omx_interfaces__srv__ExecuteTrajectory_Response *
omx_interfaces__srv__ExecuteTrajectory_Response__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  omx_interfaces__srv__ExecuteTrajectory_Response * msg = (omx_interfaces__srv__ExecuteTrajectory_Response *)allocator.allocate(sizeof(omx_interfaces__srv__ExecuteTrajectory_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(omx_interfaces__srv__ExecuteTrajectory_Response));
  bool success = omx_interfaces__srv__ExecuteTrajectory_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
omx_interfaces__srv__ExecuteTrajectory_Response__destroy(omx_interfaces__srv__ExecuteTrajectory_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    omx_interfaces__srv__ExecuteTrajectory_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__init(omx_interfaces__srv__ExecuteTrajectory_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  omx_interfaces__srv__ExecuteTrajectory_Response * data = NULL;

  if (size) {
    data = (omx_interfaces__srv__ExecuteTrajectory_Response *)allocator.zero_allocate(size, sizeof(omx_interfaces__srv__ExecuteTrajectory_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = omx_interfaces__srv__ExecuteTrajectory_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        omx_interfaces__srv__ExecuteTrajectory_Response__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__fini(omx_interfaces__srv__ExecuteTrajectory_Response__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      omx_interfaces__srv__ExecuteTrajectory_Response__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

omx_interfaces__srv__ExecuteTrajectory_Response__Sequence *
omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  omx_interfaces__srv__ExecuteTrajectory_Response__Sequence * array = (omx_interfaces__srv__ExecuteTrajectory_Response__Sequence *)allocator.allocate(sizeof(omx_interfaces__srv__ExecuteTrajectory_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__destroy(omx_interfaces__srv__ExecuteTrajectory_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__are_equal(const omx_interfaces__srv__ExecuteTrajectory_Response__Sequence * lhs, const omx_interfaces__srv__ExecuteTrajectory_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!omx_interfaces__srv__ExecuteTrajectory_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__copy(
  const omx_interfaces__srv__ExecuteTrajectory_Response__Sequence * input,
  omx_interfaces__srv__ExecuteTrajectory_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(omx_interfaces__srv__ExecuteTrajectory_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    omx_interfaces__srv__ExecuteTrajectory_Response * data =
      (omx_interfaces__srv__ExecuteTrajectory_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!omx_interfaces__srv__ExecuteTrajectory_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          omx_interfaces__srv__ExecuteTrajectory_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!omx_interfaces__srv__ExecuteTrajectory_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `info`
#include "service_msgs/msg/detail/service_event_info__functions.h"
// Member `request`
// Member `response`
// already included above
// #include "omx_interfaces/srv/detail/execute_trajectory__functions.h"

bool
omx_interfaces__srv__ExecuteTrajectory_Event__init(omx_interfaces__srv__ExecuteTrajectory_Event * msg)
{
  if (!msg) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__init(&msg->info)) {
    omx_interfaces__srv__ExecuteTrajectory_Event__fini(msg);
    return false;
  }
  // request
  if (!omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__init(&msg->request, 0)) {
    omx_interfaces__srv__ExecuteTrajectory_Event__fini(msg);
    return false;
  }
  // response
  if (!omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__init(&msg->response, 0)) {
    omx_interfaces__srv__ExecuteTrajectory_Event__fini(msg);
    return false;
  }
  return true;
}

void
omx_interfaces__srv__ExecuteTrajectory_Event__fini(omx_interfaces__srv__ExecuteTrajectory_Event * msg)
{
  if (!msg) {
    return;
  }
  // info
  service_msgs__msg__ServiceEventInfo__fini(&msg->info);
  // request
  omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__fini(&msg->request);
  // response
  omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__fini(&msg->response);
}

bool
omx_interfaces__srv__ExecuteTrajectory_Event__are_equal(const omx_interfaces__srv__ExecuteTrajectory_Event * lhs, const omx_interfaces__srv__ExecuteTrajectory_Event * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__are_equal(
      &(lhs->info), &(rhs->info)))
  {
    return false;
  }
  // request
  if (!omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__are_equal(
      &(lhs->request), &(rhs->request)))
  {
    return false;
  }
  // response
  if (!omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__are_equal(
      &(lhs->response), &(rhs->response)))
  {
    return false;
  }
  return true;
}

bool
omx_interfaces__srv__ExecuteTrajectory_Event__copy(
  const omx_interfaces__srv__ExecuteTrajectory_Event * input,
  omx_interfaces__srv__ExecuteTrajectory_Event * output)
{
  if (!input || !output) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__copy(
      &(input->info), &(output->info)))
  {
    return false;
  }
  // request
  if (!omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__copy(
      &(input->request), &(output->request)))
  {
    return false;
  }
  // response
  if (!omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__copy(
      &(input->response), &(output->response)))
  {
    return false;
  }
  return true;
}

omx_interfaces__srv__ExecuteTrajectory_Event *
omx_interfaces__srv__ExecuteTrajectory_Event__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  omx_interfaces__srv__ExecuteTrajectory_Event * msg = (omx_interfaces__srv__ExecuteTrajectory_Event *)allocator.allocate(sizeof(omx_interfaces__srv__ExecuteTrajectory_Event), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(omx_interfaces__srv__ExecuteTrajectory_Event));
  bool success = omx_interfaces__srv__ExecuteTrajectory_Event__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
omx_interfaces__srv__ExecuteTrajectory_Event__destroy(omx_interfaces__srv__ExecuteTrajectory_Event * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    omx_interfaces__srv__ExecuteTrajectory_Event__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
omx_interfaces__srv__ExecuteTrajectory_Event__Sequence__init(omx_interfaces__srv__ExecuteTrajectory_Event__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  omx_interfaces__srv__ExecuteTrajectory_Event * data = NULL;

  if (size) {
    data = (omx_interfaces__srv__ExecuteTrajectory_Event *)allocator.zero_allocate(size, sizeof(omx_interfaces__srv__ExecuteTrajectory_Event), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = omx_interfaces__srv__ExecuteTrajectory_Event__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        omx_interfaces__srv__ExecuteTrajectory_Event__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
omx_interfaces__srv__ExecuteTrajectory_Event__Sequence__fini(omx_interfaces__srv__ExecuteTrajectory_Event__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      omx_interfaces__srv__ExecuteTrajectory_Event__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

omx_interfaces__srv__ExecuteTrajectory_Event__Sequence *
omx_interfaces__srv__ExecuteTrajectory_Event__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  omx_interfaces__srv__ExecuteTrajectory_Event__Sequence * array = (omx_interfaces__srv__ExecuteTrajectory_Event__Sequence *)allocator.allocate(sizeof(omx_interfaces__srv__ExecuteTrajectory_Event__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = omx_interfaces__srv__ExecuteTrajectory_Event__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
omx_interfaces__srv__ExecuteTrajectory_Event__Sequence__destroy(omx_interfaces__srv__ExecuteTrajectory_Event__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    omx_interfaces__srv__ExecuteTrajectory_Event__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
omx_interfaces__srv__ExecuteTrajectory_Event__Sequence__are_equal(const omx_interfaces__srv__ExecuteTrajectory_Event__Sequence * lhs, const omx_interfaces__srv__ExecuteTrajectory_Event__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!omx_interfaces__srv__ExecuteTrajectory_Event__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
omx_interfaces__srv__ExecuteTrajectory_Event__Sequence__copy(
  const omx_interfaces__srv__ExecuteTrajectory_Event__Sequence * input,
  omx_interfaces__srv__ExecuteTrajectory_Event__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(omx_interfaces__srv__ExecuteTrajectory_Event);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    omx_interfaces__srv__ExecuteTrajectory_Event * data =
      (omx_interfaces__srv__ExecuteTrajectory_Event *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!omx_interfaces__srv__ExecuteTrajectory_Event__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          omx_interfaces__srv__ExecuteTrajectory_Event__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!omx_interfaces__srv__ExecuteTrajectory_Event__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
