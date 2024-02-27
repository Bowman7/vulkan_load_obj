#ifndef FUNC_H
#define FUNC_H

//check reuired extension is present or not
void checkRequiredExtension(std::vector<VkExtensionProperties>extensions,const char** glfwExtensions,uint32_t glfwExtensionCount)
{
  for(auto i = 0;i<glfwExtensionCount;i++)
    {
      bool found =false;
      for(const VkExtensionProperties& extension: extensions)
	{
	  if(strcmp(glfwExtensions[i],extension.extensionName))
	    {
	      found =true;
	    }
	}
      if(!found)
	{
	  throw std::runtime_error("MISSING VULKAN EXTENSION.\n");
	}
    }
  std::cout<<"EXTENSIONS ARE PRESENT"<<std::endl;
}

//message callback function

std::vector<const char*>getRequiredExtensions()
{
  uint32_t glfwExtensionCount = 0;
  const char** glfwExtensions;
  glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

  std::vector<const char*>extensions(glfwExtensions,glfwExtensions+glfwExtensionCount);
  if(enableValidationLayers)
    {
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
  return extensions;
}

//create vkdebuguitlsmessengerext object
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}
//destroy vkdebugutilsmessengerext object

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

#endif
