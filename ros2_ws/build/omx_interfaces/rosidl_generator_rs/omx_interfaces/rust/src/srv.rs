#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};




// Corresponds to omx_interfaces__srv__GenerateTrajectory_Request

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct GenerateTrajectory_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub qi: Vec<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub qf: Vec<f64>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub ts: f64,

}



impl Default for GenerateTrajectory_Request {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::GenerateTrajectory_Request::default())
  }
}

impl rosidl_runtime_rs::Message for GenerateTrajectory_Request {
  type RmwMsg = super::srv::rmw::GenerateTrajectory_Request;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        qi: msg.qi.into(),
        qf: msg.qf.into(),
        ts: msg.ts,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        qi: msg.qi.as_slice().into(),
        qf: msg.qf.as_slice().into(),
      ts: msg.ts,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      qi: msg.qi
          .into_iter()
          .collect(),
      qf: msg.qf
          .into_iter()
          .collect(),
      ts: msg.ts,
    }
  }
}


// Corresponds to omx_interfaces__srv__GenerateTrajectory_Response

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct GenerateTrajectory_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub trajectory: trajectory_msgs::msg::JointTrajectory,


    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,


    // This member is not documented.
    #[allow(missing_docs)]
    pub message: std::string::String,

}



impl Default for GenerateTrajectory_Response {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::GenerateTrajectory_Response::default())
  }
}

impl rosidl_runtime_rs::Message for GenerateTrajectory_Response {
  type RmwMsg = super::srv::rmw::GenerateTrajectory_Response;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        trajectory: trajectory_msgs::msg::JointTrajectory::into_rmw_message(std::borrow::Cow::Owned(msg.trajectory)).into_owned(),
        success: msg.success,
        message: msg.message.as_str().into(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        trajectory: trajectory_msgs::msg::JointTrajectory::into_rmw_message(std::borrow::Cow::Borrowed(&msg.trajectory)).into_owned(),
      success: msg.success,
        message: msg.message.as_str().into(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      trajectory: trajectory_msgs::msg::JointTrajectory::from_rmw_message(msg.trajectory),
      success: msg.success,
      message: msg.message.to_string(),
    }
  }
}


// Corresponds to omx_interfaces__srv__ExecuteTrajectory_Request

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ExecuteTrajectory_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub trajectory: trajectory_msgs::msg::JointTrajectory,

}



impl Default for ExecuteTrajectory_Request {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::ExecuteTrajectory_Request::default())
  }
}

impl rosidl_runtime_rs::Message for ExecuteTrajectory_Request {
  type RmwMsg = super::srv::rmw::ExecuteTrajectory_Request;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        trajectory: trajectory_msgs::msg::JointTrajectory::into_rmw_message(std::borrow::Cow::Owned(msg.trajectory)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        trajectory: trajectory_msgs::msg::JointTrajectory::into_rmw_message(std::borrow::Cow::Borrowed(&msg.trajectory)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      trajectory: trajectory_msgs::msg::JointTrajectory::from_rmw_message(msg.trajectory),
    }
  }
}


// Corresponds to omx_interfaces__srv__ExecuteTrajectory_Response

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ExecuteTrajectory_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,


    // This member is not documented.
    #[allow(missing_docs)]
    pub message: std::string::String,

}



impl Default for ExecuteTrajectory_Response {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::ExecuteTrajectory_Response::default())
  }
}

impl rosidl_runtime_rs::Message for ExecuteTrajectory_Response {
  type RmwMsg = super::srv::rmw::ExecuteTrajectory_Response;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        success: msg.success,
        message: msg.message.as_str().into(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      success: msg.success,
        message: msg.message.as_str().into(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      success: msg.success,
      message: msg.message.to_string(),
    }
  }
}


// Corresponds to omx_interfaces__srv__SetGripper_Request

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct SetGripper_Request {

    // This member is not documented.
    #[allow(missing_docs)]
    pub position: f64,

}



impl Default for SetGripper_Request {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::SetGripper_Request::default())
  }
}

impl rosidl_runtime_rs::Message for SetGripper_Request {
  type RmwMsg = super::srv::rmw::SetGripper_Request;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        position: msg.position,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      position: msg.position,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      position: msg.position,
    }
  }
}


// Corresponds to omx_interfaces__srv__SetGripper_Response

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct SetGripper_Response {

    // This member is not documented.
    #[allow(missing_docs)]
    pub success: bool,


    // This member is not documented.
    #[allow(missing_docs)]
    pub message: std::string::String,

}



impl Default for SetGripper_Response {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::SetGripper_Response::default())
  }
}

impl rosidl_runtime_rs::Message for SetGripper_Response {
  type RmwMsg = super::srv::rmw::SetGripper_Response;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        success: msg.success,
        message: msg.message.as_str().into(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      success: msg.success,
        message: msg.message.as_str().into(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      success: msg.success,
      message: msg.message.to_string(),
    }
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


