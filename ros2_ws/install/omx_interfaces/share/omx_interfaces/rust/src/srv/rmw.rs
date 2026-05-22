#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



#[link(name = "omx_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__omx_interfaces__srv__GenerateTrajectory_Request() -> *const std::ffi::c_void;
}

#[link(name = "omx_interfaces__rosidl_generator_c")]
extern "C" {
    fn omx_interfaces__srv__GenerateTrajectory_Request__init(msg: *mut GenerateTrajectory_Request) -> bool;
    fn omx_interfaces__srv__GenerateTrajectory_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<GenerateTrajectory_Request>, size: usize) -> bool;
    fn omx_interfaces__srv__GenerateTrajectory_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<GenerateTrajectory_Request>);
    fn omx_interfaces__srv__GenerateTrajectory_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<GenerateTrajectory_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<GenerateTrajectory_Request>) -> bool;
}

// Corresponds to omx_interfaces__srv__GenerateTrajectory_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct GenerateTrajectory_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub waypoints: rosidl_runtime_rs::Sequence<trajectory_msgs::msg::rmw::JointTrajectoryPoint>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub joint_names: rosidl_runtime_rs::Sequence<rosidl_runtime_rs::String>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub duration: f64,

}



impl Default for GenerateTrajectory_Request {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !omx_interfaces__srv__GenerateTrajectory_Request__init(&mut msg as *mut _) {
        panic!("Call to omx_interfaces__srv__GenerateTrajectory_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for GenerateTrajectory_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__GenerateTrajectory_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__GenerateTrajectory_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__GenerateTrajectory_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for GenerateTrajectory_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for GenerateTrajectory_Request where Self: Sized {
  const TYPE_NAME: &'static str = "omx_interfaces/srv/GenerateTrajectory_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__omx_interfaces__srv__GenerateTrajectory_Request() }
  }
}


#[link(name = "omx_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__omx_interfaces__srv__GenerateTrajectory_Response() -> *const std::ffi::c_void;
}

#[link(name = "omx_interfaces__rosidl_generator_c")]
extern "C" {
    fn omx_interfaces__srv__GenerateTrajectory_Response__init(msg: *mut GenerateTrajectory_Response) -> bool;
    fn omx_interfaces__srv__GenerateTrajectory_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<GenerateTrajectory_Response>, size: usize) -> bool;
    fn omx_interfaces__srv__GenerateTrajectory_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<GenerateTrajectory_Response>);
    fn omx_interfaces__srv__GenerateTrajectory_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<GenerateTrajectory_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<GenerateTrajectory_Response>) -> bool;
}

// Corresponds to omx_interfaces__srv__GenerateTrajectory_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct GenerateTrajectory_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub trajectory: trajectory_msgs::msg::rmw::JointTrajectory,


    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,


    // This member is not documented.
    #[allow(missing_docs)]
    pub message: rosidl_runtime_rs::String,

}



impl Default for GenerateTrajectory_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !omx_interfaces__srv__GenerateTrajectory_Response__init(&mut msg as *mut _) {
        panic!("Call to omx_interfaces__srv__GenerateTrajectory_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for GenerateTrajectory_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__GenerateTrajectory_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__GenerateTrajectory_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__GenerateTrajectory_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for GenerateTrajectory_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for GenerateTrajectory_Response where Self: Sized {
  const TYPE_NAME: &'static str = "omx_interfaces/srv/GenerateTrajectory_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__omx_interfaces__srv__GenerateTrajectory_Response() }
  }
}


#[link(name = "omx_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__omx_interfaces__srv__ExecuteTrajectory_Request() -> *const std::ffi::c_void;
}

#[link(name = "omx_interfaces__rosidl_generator_c")]
extern "C" {
    fn omx_interfaces__srv__ExecuteTrajectory_Request__init(msg: *mut ExecuteTrajectory_Request) -> bool;
    fn omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<ExecuteTrajectory_Request>, size: usize) -> bool;
    fn omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<ExecuteTrajectory_Request>);
    fn omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<ExecuteTrajectory_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<ExecuteTrajectory_Request>) -> bool;
}

// Corresponds to omx_interfaces__srv__ExecuteTrajectory_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ExecuteTrajectory_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub trajectory: trajectory_msgs::msg::rmw::JointTrajectory,

}



impl Default for ExecuteTrajectory_Request {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !omx_interfaces__srv__ExecuteTrajectory_Request__init(&mut msg as *mut _) {
        panic!("Call to omx_interfaces__srv__ExecuteTrajectory_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for ExecuteTrajectory_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__ExecuteTrajectory_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for ExecuteTrajectory_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for ExecuteTrajectory_Request where Self: Sized {
  const TYPE_NAME: &'static str = "omx_interfaces/srv/ExecuteTrajectory_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__omx_interfaces__srv__ExecuteTrajectory_Request() }
  }
}


#[link(name = "omx_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__omx_interfaces__srv__ExecuteTrajectory_Response() -> *const std::ffi::c_void;
}

#[link(name = "omx_interfaces__rosidl_generator_c")]
extern "C" {
    fn omx_interfaces__srv__ExecuteTrajectory_Response__init(msg: *mut ExecuteTrajectory_Response) -> bool;
    fn omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<ExecuteTrajectory_Response>, size: usize) -> bool;
    fn omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<ExecuteTrajectory_Response>);
    fn omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<ExecuteTrajectory_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<ExecuteTrajectory_Response>) -> bool;
}

// Corresponds to omx_interfaces__srv__ExecuteTrajectory_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ExecuteTrajectory_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,


    // This member is not documented.
    #[allow(missing_docs)]
    pub message: rosidl_runtime_rs::String,

}



impl Default for ExecuteTrajectory_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !omx_interfaces__srv__ExecuteTrajectory_Response__init(&mut msg as *mut _) {
        panic!("Call to omx_interfaces__srv__ExecuteTrajectory_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for ExecuteTrajectory_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__ExecuteTrajectory_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for ExecuteTrajectory_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for ExecuteTrajectory_Response where Self: Sized {
  const TYPE_NAME: &'static str = "omx_interfaces/srv/ExecuteTrajectory_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__omx_interfaces__srv__ExecuteTrajectory_Response() }
  }
}


#[link(name = "omx_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__omx_interfaces__srv__SetGripper_Request() -> *const std::ffi::c_void;
}

#[link(name = "omx_interfaces__rosidl_generator_c")]
extern "C" {
    fn omx_interfaces__srv__SetGripper_Request__init(msg: *mut SetGripper_Request) -> bool;
    fn omx_interfaces__srv__SetGripper_Request__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<SetGripper_Request>, size: usize) -> bool;
    fn omx_interfaces__srv__SetGripper_Request__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<SetGripper_Request>);
    fn omx_interfaces__srv__SetGripper_Request__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<SetGripper_Request>, out_seq: *mut rosidl_runtime_rs::Sequence<SetGripper_Request>) -> bool;
}

// Corresponds to omx_interfaces__srv__SetGripper_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct SetGripper_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub position: f64,

}



impl Default for SetGripper_Request {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !omx_interfaces__srv__SetGripper_Request__init(&mut msg as *mut _) {
        panic!("Call to omx_interfaces__srv__SetGripper_Request__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for SetGripper_Request {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__SetGripper_Request__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__SetGripper_Request__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__SetGripper_Request__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for SetGripper_Request {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for SetGripper_Request where Self: Sized {
  const TYPE_NAME: &'static str = "omx_interfaces/srv/SetGripper_Request";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__omx_interfaces__srv__SetGripper_Request() }
  }
}


#[link(name = "omx_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__omx_interfaces__srv__SetGripper_Response() -> *const std::ffi::c_void;
}

#[link(name = "omx_interfaces__rosidl_generator_c")]
extern "C" {
    fn omx_interfaces__srv__SetGripper_Response__init(msg: *mut SetGripper_Response) -> bool;
    fn omx_interfaces__srv__SetGripper_Response__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<SetGripper_Response>, size: usize) -> bool;
    fn omx_interfaces__srv__SetGripper_Response__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<SetGripper_Response>);
    fn omx_interfaces__srv__SetGripper_Response__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<SetGripper_Response>, out_seq: *mut rosidl_runtime_rs::Sequence<SetGripper_Response>) -> bool;
}

// Corresponds to omx_interfaces__srv__SetGripper_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct SetGripper_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,


    // This member is not documented.
    #[allow(missing_docs)]
    pub message: rosidl_runtime_rs::String,

}



impl Default for SetGripper_Response {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !omx_interfaces__srv__SetGripper_Response__init(&mut msg as *mut _) {
        panic!("Call to omx_interfaces__srv__SetGripper_Response__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for SetGripper_Response {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__SetGripper_Response__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__SetGripper_Response__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { omx_interfaces__srv__SetGripper_Response__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for SetGripper_Response {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for SetGripper_Response where Self: Sized {
  const TYPE_NAME: &'static str = "omx_interfaces/srv/SetGripper_Response";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__omx_interfaces__srv__SetGripper_Response() }
  }
}






#[link(name = "omx_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__omx_interfaces__srv__GenerateTrajectory() -> *const std::ffi::c_void;
}

// Corresponds to omx_interfaces__srv__GenerateTrajectory
#[allow(missing_docs, non_camel_case_types)]
pub struct GenerateTrajectory;

impl rosidl_runtime_rs::Service for GenerateTrajectory {
    type Request = GenerateTrajectory_Request;
    type Response = GenerateTrajectory_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__omx_interfaces__srv__GenerateTrajectory() }
    }
}




#[link(name = "omx_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__omx_interfaces__srv__ExecuteTrajectory() -> *const std::ffi::c_void;
}

// Corresponds to omx_interfaces__srv__ExecuteTrajectory
#[allow(missing_docs, non_camel_case_types)]
pub struct ExecuteTrajectory;

impl rosidl_runtime_rs::Service for ExecuteTrajectory {
    type Request = ExecuteTrajectory_Request;
    type Response = ExecuteTrajectory_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__omx_interfaces__srv__ExecuteTrajectory() }
    }
}




#[link(name = "omx_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__omx_interfaces__srv__SetGripper() -> *const std::ffi::c_void;
}

// Corresponds to omx_interfaces__srv__SetGripper
#[allow(missing_docs, non_camel_case_types)]
pub struct SetGripper;

impl rosidl_runtime_rs::Service for SetGripper {
    type Request = SetGripper_Request;
    type Response = SetGripper_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__omx_interfaces__srv__SetGripper() }
    }
}


