#define GLFW_INCLUDE_VULKAN
#include<GLFW/glfw3.h>
#define GLM_FORCE_RADIANS//use radians as arguments
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_ENABLE_EXPERIMENTAL

#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>//gen model transformations like glm::rotate,glm::lookAt,glm::perspective
#include<chrono>//exposes fn to do precise timekeeping, rotates per sec regardless of framerate
#define STB_IMAGE_IMPLEMENTATION
#include<stb_image.h>

//tinyobj
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include<unordered_map>

#include <glm/gtx/hash.hpp>

#include<array>
#include<iostream>
#include<stdexcept>
#include<cstdlib>
#include<vector>
#include<cstring>
#include<optional>
#include<set>
#include<cstdint>//uint32_t
#include<limits>//std::numeric_limits<uint32_t>
#include<algorithm>//std::clamp
#include<fstream>

//UNIFORM BUFFER
struct UniformBufferObject
{
  //glm::vec2 foo;
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};
//Vertex data
struct Vertex
{
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  //first struct to bind
  static VkVertexInputBindingDescription getBindingDescription()
  {
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    
    return bindingDescription;
  }
  //second struct to bind
  static std::array<VkVertexInputAttributeDescription,3> getAttributeDescriptions()
  {
    std::array<VkVertexInputAttributeDescription,3> attributeDescriptions{};
    //for pos
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex,pos);
    //for color
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex,color);
    //for texture
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Vertex,texCoord);

    return attributeDescriptions;
  }

  bool operator==(const Vertex& other) const
  {
    return pos ==other.pos && color == other.color && texCoord == other.texCoord;
  }
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^
                   (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
                   (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}

//pos and colors
/*const std::vector<Vertex> vertices ={
  {{0.0f,-0.5f},{1.0f,0.0f,0.0f}},
  {{0.5f,0.5f},{0.0f,1.0f,0.0f}},
  {{-0.5f,0.5f},{0.0f,0.0f,1.0f}}
  };
const std::vector<Vertex> vertices ={
  {{-0.5f,-0.5f,0.0f},{1.0f,0.0f,0.0f},{1.0f,0.0f}},
  {{0.5f,-0.5f,0.0f},{0.0f,1.0f,0.0f},{0.0f,0.0f}},
  {{0.5f,0.5f,0.0f},{0.0f,0.0f,1.0f},{0.0f,1.0f}},
  {{-0.5f,0.5f,0.0f},{1.0f,1.0f,1.0f},{1.0f,1.0f}},

   {{-0.5f,-0.5f,-0.5f},{1.0f,0.0f,0.0f},{1.0f,0.0f}},
  {{0.5f,-0.5f,-0.5f},{0.0f,1.0f,0.0f},{0.0f,0.0f}},
  {{0.5f,0.5f,-0.5f},{0.0f,0.0f,1.0f},{0.0f,1.0f}},
  {{-0.5f,0.5f,-0.5f},{1.0f,1.0f,1.0f},{1.0f,1.0f}}
  
};
//indices
const std::vector<uint16_t> indices={
  0,1,2,2,3,0,
  4,5,6,6,7,4,
  0,4,3,3,7,4,
  7,3,2,2,6,7,
  6,2,5,5,1,2,
  1,5,0,0,4,5
};
*/
//defining how many frames should run concurrently
const int MAX_FRAMES_IN_FLIGHT = 2;
const uint32_t WIDTH =800;
const uint32_t HEIGHT = 600;

//include MODEL PATHS

const std::string MODEL_PATH ="models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";

//enable validation layers

const std::vector<const char*>validationLayers = {
  "VK_LAYER_KHRONOS_validation"
};
//on or off
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;

#endif
//QUEUE FAMILY INDIES

struct QueueFamilyIndices
{
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t>presentFamily;
  bool isComplete()
  {
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};
//SWAP CHAIN SUPPORT DETAILS
struct SwapChainSupportDetails
{
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

//swap chain
const std::vector<const char*> deviceExtensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
//functions
#include"func.h"

class HelloTriangleApplication
{
public:
  void run()
  {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }
private:
  void initWindow()
  {
    //init glfw
    glfwInit();
    //tell glfw not to create an opengl context
    glfwWindowHint(GLFW_CLIENT_API,GLFW_NO_API);
    //glfwWindowHint(GLFW_RESIZABLE,GLFW_FALSE);//no resizable window

    window = glfwCreateWindow(WIDTH,HEIGHT,"Vulkan",nullptr,nullptr);
    glfwSetWindowUserPointer(window,this);
    //resize callback
    glfwSetFramebufferSizeCallback(window,framebufferResizeCallback);
    
  }
  static void framebufferResizeCallback(GLFWwindow* window,int width,int height)
  {
    auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
  }
  void initVulkan()
  {
    createInstance();
    setupDebugMessenger();
    createSurface();//create surface
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();//render pass
    createDescriptorSetLayout();//descriptor set layout
    createGraphicsPipeline();//create graphics pipeline
    
    createCommandPool();
    createColorResources();
    createDepthResources();
    createFramebuffers();//create frame buffer
    
    createTextureImage();
    createTextureImageView();
    createTextureSampler();

    loadModel();
    
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffer();
    createSyncObjects();
    
    
  }

  //colorResources
  void createColorResources()
  {
    VkFormat colorFormat = swapChainImageFormat;

    createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat,
		VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		colorImage, colorImageMemory);
    colorImageView = createImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
  }
  //Func to see max no. of samples usable
  VkSampleCountFlagBits getMaxUsableSampleCountBits()
  {
    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice,&physicalDeviceProperties);

    VkSampleCountFlags count = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    if(count & VK_SAMPLE_COUNT_64_BIT){return VK_SAMPLE_COUNT_64_BIT;}
    if(count & VK_SAMPLE_COUNT_32_BIT){return VK_SAMPLE_COUNT_32_BIT;}
    if(count & VK_SAMPLE_COUNT_16_BIT){return VK_SAMPLE_COUNT_16_BIT;}
    if(count & VK_SAMPLE_COUNT_8_BIT){return VK_SAMPLE_COUNT_8_BIT;}
    if(count & VK_SAMPLE_COUNT_4_BIT){return VK_SAMPLE_COUNT_4_BIT;}
    if(count & VK_SAMPLE_COUNT_2_BIT){return VK_SAMPLE_COUNT_2_BIT;}

    return VK_SAMPLE_COUNT_1_BIT;
  }
  //LOAD MODEL
  void loadModel()
  {
    tinyobj::attrib_t  attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn,err;

    //load
    if(!tinyobj::LoadObj(&attrib,&shapes,&materials,&warn,&err,MODEL_PATH.c_str()))
      {
	throw std::runtime_error(warn+err);
      }

    std::unordered_map<Vertex ,uint32_t> uniqueVertices{};
    for(const auto& shape : shapes)
      {
	for(const auto& index : shape.mesh.indices)
	  {
	    Vertex vertex{};

	    vertex.pos = {
	      attrib.vertices[ 3 * index.vertex_index + 0],
	      attrib.vertices[ 3 * index.vertex_index + 1],
	      attrib.vertices[ 3 * index.vertex_index + 2]
	    };

	    vertex.texCoord = {
	      attrib.texcoords[2 * index.texcoord_index + 0],
	      1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
	    };

	    vertex.color = {1.0f,1.0f,1.0f};

	    if(uniqueVertices.count(vertex) == 0)
	      {
		uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
		vertices.push_back(vertex);
	      }
	    indices.push_back(uniqueVertices[vertex]);
	  }
      }
    
  }
  //check for stencil component
  bool hasStencilComponent(VkFormat format)
  {
    return format==VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
  }
  //helper function to only find if required formats are present
  VkFormat findDepthFormat()
  {
    return findSupportedFormat(
			       {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
			       VK_IMAGE_TILING_OPTIMAL,
			       VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
			       );
  }
  //check for supported formats for features
  VkFormat findSupportedFormat(const std::vector<VkFormat>&candidates,VkImageTiling tiling,VkFormatFeatureFlags features)
  {
    for(VkFormat format : candidates)
      {
	VkFormatProperties props;
	vkGetPhysicalDeviceFormatProperties(physicalDevice,format,&props);

	if(tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
	  {
	    return format;
	  }
	else if(tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
	  {
	    return format;
	  }
      }
    throw std::runtime_error("Failed to find supported format");
  }
  //CREATE DEPTH RESOURCES
  void createDepthResources()
  {
    VkFormat depthFormat = findDepthFormat();

    createImage(swapChainExtent.width, swapChainExtent.height,1,msaaSamples,depthFormat,VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		depthImage,depthImageMemory);
    depthImageView = createImageView(depthImage,depthFormat,VK_IMAGE_ASPECT_DEPTH_BIT,1);

    //explicitly transitioning depth images
    transitionImageLayout(depthImage,depthFormat,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,1);
    
  }
  
  //textures are usually accessed through samplers
  void createTextureSampler()
  {
 
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice,&properties);
    
    VkSamplerCreateInfo samplerInfo{};

    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter  = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT ;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT ; 
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT ; 
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;//static_cast<float>(mipLevels/2)
    samplerInfo.maxLod = static_cast<float>(mipLevels);
    
    if(vkCreateSampler(device,&samplerInfo,nullptr,&textureSampler) != VK_SUCCESS)
      {
	throw std::runtime_error("Failed to create texture sampler!");
      }
    
  }
  //abstract creating image view into a single fn
  VkImageView createImageView(VkImage image,VkFormat format,VkImageAspectFlags aspectFlags,uint32_t mipLevels)
  {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    viewInfo.subresourceRange.aspectMask = aspectFlags;

    VkImageView imageView;
    if(vkCreateImageView(device,&viewInfo,nullptr,&imageView) != VK_SUCCESS)
      {
	throw std::runtime_error("Failed to create image views!");
      }
    return imageView;
  }
  //create image view for the texture
  void createTextureImageView()
  {
    textureImageView = createImageView(textureImage,VK_FORMAT_R8G8B8A8_SRGB,VK_IMAGE_ASPECT_COLOR_BIT,mipLevels);

  }
  //HELPER function to copy buffer to iamge
  void copyBufferToImage(VkBuffer buffer,VkImage image,uint32_t width,uint32_t height)
  {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferImageCopy region{};

    region.bufferOffset = 0;
    region.bufferRowLength =0 ;
    region.bufferImageHeight =0 ;
    region.imageSubresource.aspectMask =VK_IMAGE_ASPECT_COLOR_BIT ;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0,0,0};
    region.imageExtent = {width,height,1};

    vkCmdCopyBufferToImage(commandBuffer,
			   buffer,
			   image,
			   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			   1,
			   &region);
    
    endSingleTimeCommands(commandBuffer);
  }
  //Handle layout transition
  void transitionImageLayout(VkImage image,VkFormat format,VkImageLayout oldLayout,VkImageLayout newLayout,uint32_t mipLevels)
  {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();
    
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    if(newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
      {
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	if(hasStencilComponent(format))
	  {
	    barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
	  }
      }
    else
      {
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      }

    //transition barrier masks

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if(oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
      {
        barrier.srcAccessMask = 0;
	barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

	sourceStage      = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      }
    else if(oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
      {
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

	sourceStage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
	destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      }
    else if(oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
      {
	barrier.srcAccessMask = 0;
	barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

	sourceStage      = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
      }

    else
      {
	throw std::runtime_error("unsupported layout transition");
      }
    
    

    vkCmdPipelineBarrier(commandBuffer,
			 sourceStage,destinationStage,
			 0,
			 0,nullptr,
			 0,nullptr,
			 1,&barrier);
    endSingleTimeCommands(commandBuffer);
  }
  //CREATE TEXTURE IMAGE
  void createTextureImage()
  {
    int texWidth,texHeight,texChannels;
    //stbi_uc* pixels = stbi_load("textures/texture.jpg",&texWidth,&texHeight,&texChannels,STBI_rgb_alpha);
    //for model path
    stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(),&texWidth,&texHeight,&texChannels,STBI_rgb_alpha);
    
    VkDeviceSize imageSize = texWidth * texHeight * 4;
    mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth,texHeight))))+1;
    if(!pixels)
      {
	throw std::runtime_error("Could not load image!");
      }
    //STAGING BUFFER
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    createBuffer(imageSize,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		 stagingBuffer,stagingBufferMemory);
    void* data;
    vkMapMemory(device,stagingBufferMemory,0,imageSize,0,&data);
    memcpy(data,pixels,static_cast<size_t>(imageSize));
    vkUnmapMemory(device,stagingBufferMemory);

    stbi_image_free(pixels);//cleanup

    createImage(texWidth, texHeight, mipLevels,VK_SAMPLE_COUNT_1_BIT,VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		textureImage, textureImageMemory);

    transitionImageLayout(textureImage,VK_FORMAT_R8G8B8A8_SRGB,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,mipLevels);

    copyBufferToImage(stagingBuffer,textureImage,static_cast<uint32_t>(texWidth),static_cast<uint32_t>(texHeight));

    //transitionImageLayout(textureImage,VK_FORMAT_R8G8B8A8_SRGB,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,mipLevels);
    generateMipmaps(textureImage,VK_FORMAT_R8G8B8A8_SRGB,texWidth,texHeight,mipLevels);

    vkDestroyBuffer(device,stagingBuffer,nullptr);
    vkFreeMemory(device,stagingBufferMemory,nullptr);
    
  }
  void generateMipmaps(VkImage image,VkFormat imageFormat,int32_t texWidth,int32_t texHeight,uint32_t mipLevels)
  {
    // Check if image format supports linear blitting
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);

    if(!(formatProperties.optimalTilingFeatures && VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
      {
	throw std::runtime_error("texture image format does not support linear blitting");
      }
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask =  VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;
    

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for(uint32_t i = 1;i<mipLevels;i++)
      {
	barrier.subresourceRange.baseMipLevel  = i-1 ;
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

	vkCmdPipelineBarrier(commandBuffer,
			     VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,
			     0,nullptr,
			     0,nullptr,
			     1,&barrier
			     );
	VkImageBlit blit{};
	blit.srcOffsets[0] = {0,0,0};
	blit.srcOffsets[1] = {mipWidth,mipHeight,1};
	blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	blit.srcSubresource.mipLevel = i-1;
	blit.srcSubresource.baseArrayLayer = 0;
	blit.srcSubresource.layerCount = 1;
	blit.dstOffsets[0] = {0,0,0};
	blit.dstOffsets[1] = {mipWidth>1?mipWidth/2:1,mipHeight>1?mipHeight/2:1,1};

	blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	blit.dstSubresource.mipLevel = i;
	blit.dstSubresource.baseArrayLayer = 0;
	blit.dstSubresource.layerCount = 1;

	vkCmdBlitImage(commandBuffer,
		       image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		       image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		       1,&blit,
		       VK_FILTER_LINEAR
		       );

	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

	vkCmdPipelineBarrier(commandBuffer,
			     VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,
			     0,nullptr,
			     0,nullptr,
			     1,&barrier
			     );
	if(mipWidth>1) mipWidth /= 2;
	if(mipHeight>1) mipHeight /= 2;
      }
    barrier.subresourceRange.baseMipLevel  = mipLevels-1 ;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer,
			     VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,
			     0,nullptr,
			     0,nullptr,
			     1,&barrier
			     );
    
    endSingleTimeCommands(commandBuffer);

  }
  void createImage(uint32_t width, uint32_t height,uint32_t mipLevels,VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
		  VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
  {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = numSamples;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;


    if(vkCreateImage(device,&imageInfo,nullptr,&image) != VK_SUCCESS)
      {
	throw std::runtime_error("Could not create image info!");
      }
    //memory requirements
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device,image,&memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,
						properties);
    if(vkAllocateMemory(device,&allocInfo,nullptr,&imageMemory) != VK_SUCCESS)
      {
	throw std::runtime_error("Failed to allocate image memory");
      }
    vkBindImageMemory(device,image,imageMemory,0);
  }
  //CREATE DESCRIPTOR SETS
  void createDescriptorSets()
  {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts = layouts.data();
    
    descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if(vkAllocateDescriptorSets(device,&allocInfo,descriptorSets.data()) != VK_SUCCESS)
      {
	throw std::runtime_error("Could not allocate descriptor sets!");
      }
    //populate descriptor
    for(size_t i =0 ;i<MAX_FRAMES_IN_FLIGHT;i++)
      {
	VkDescriptorBufferInfo bufferInfo{};
	bufferInfo.buffer = uniformBuffers[i];
	bufferInfo.offset = 0;
	bufferInfo.range= sizeof(UniformBufferObject);

	VkDescriptorImageInfo imageInfos{};
	imageInfos.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	imageInfos.imageView = textureImageView;
	imageInfos.sampler = textureSampler;
	//config of descriptors are updated with a fn that takes writeDescrptor set as parameter
	//for unfirom buffer descriptor
	std::array<VkWriteDescriptorSet,2> descriptorWrites{};
	descriptorWrites[0].sType= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrites[0].dstSet = descriptorSets[i];
	descriptorWrites[0].dstBinding = 0;
	descriptorWrites[0].dstArrayElement = 0;
	descriptorWrites[0].descriptorCount = 1;
	descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descriptorWrites[0].pImageInfo = nullptr;
	descriptorWrites[0].pBufferInfo = &bufferInfo;
	descriptorWrites[0].pTexelBufferView = nullptr;
	//for texture
	descriptorWrites[1].sType= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrites[1].dstSet = descriptorSets[i];
	descriptorWrites[1].dstBinding = 1;
	descriptorWrites[1].dstArrayElement = 0;
	descriptorWrites[1].descriptorCount = 1;
	descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	descriptorWrites[1].pImageInfo = &imageInfos;
	descriptorWrites[1].pBufferInfo = nullptr;
	descriptorWrites[1].pTexelBufferView = nullptr;
	
	//update
	vkUpdateDescriptorSets(device,static_cast<uint32_t>(descriptorWrites.size()),descriptorWrites.data(),0,nullptr);
	
      }
    
      
  }
  //CREATE DESCRIPTOR POOL
  void createDescriptorPool()
  {
    //specify poolsize info
    std::array<VkDescriptorPoolSize,2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    //create info
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    if(vkCreateDescriptorPool(device,&poolInfo,nullptr,&descriptorPool) != VK_SUCCESS)
      {
	throw std::runtime_error("Could not create descriptor pool!");
      }
    
    
    
  }
  //CREATE UNIFORM BUFFER
  void createUniformBuffers()
  {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for(size_t i=0;i<MAX_FRAMES_IN_FLIGHT;i++)
      {
	createBuffer(bufferSize,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		     uniformBuffers[i],uniformBuffersMemory[i]);
	vkMapMemory(device,uniformBuffersMemory[i],0,bufferSize,0,&uniformBuffersMapped[i]);
      }
  }
  //CREATING A DESRIPTOR SET LAYOUT
  void createDescriptorSetLayout()
  {
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;//only reference in vertex shader 
    uboLayoutBinding.pImmutableSamplers= nullptr;

   

    //creating an image sampler
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;//only reference in fragment bit

    std::array<VkDescriptorSetLayoutBinding,2>bindings = {uboLayoutBinding,samplerLayoutBinding}; 

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    //layoutInfo.flags =;
    layoutInfo.bindingCount =static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if(vkCreateDescriptorSetLayout(device,&layoutInfo,nullptr,&descriptorSetLayout) != VK_SUCCESS)
      {
	throw std::runtime_error("Could not create descriptor set layout!");
      }
    
    
  }
  //CREATE INDEX BUFFER
  void createIndexBuffer()
  {
    VkDeviceSize bufferSize = sizeof(indices[0])*indices.size();
    //creating a staging buffer with src flag
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    
    createBuffer(bufferSize,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		 stagingBuffer,stagingBufferMemory);
    //filling vertexBuffer
    void* data;
    vkMapMemory(device,stagingBufferMemory,0,bufferSize,0,&data);
    memcpy(data,indices.data(),(size_t)bufferSize);
    vkUnmapMemory(device,stagingBufferMemory);

    createBuffer(bufferSize,VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		 indexBuffer,indexBufferMemory);
    //transfer the data from staging buffer to actual vertex buffer
    copyBuffer(stagingBuffer,indexBuffer,bufferSize);
    //cleanup staging buffers
    vkDestroyBuffer(device,stagingBuffer,nullptr);
    vkFreeMemory(device,stagingBufferMemory,nullptr);
  }
  //HELPER FUNCTION OF RECORDING AND EXECUTING A COMMAND BUFFER
  VkCommandBuffer beginSingleTimeCommands()
  {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device,&allocInfo,&commandBuffer);

    //immediately start recording command buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer,&beginInfo);

    return commandBuffer;
  }

  void endSingleTimeCommands(VkCommandBuffer commandBuffer)
  {
    //end cmd buffer
    vkEndCommandBuffer(commandBuffer);
    //execute the command buffer to complete the transfer
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    //use queue instead if fence as we only have a single cmd buffer
    vkQueueSubmit(graphicsQueue ,1, &submitInfo,VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    //free cmd buffer
    vkFreeCommandBuffers(device,commandPool,1,&commandBuffer);
  }
  //FN TO COPY data from staging buffer to actual vertex buffer
  void copyBuffer(VkBuffer srcBuffer,VkBuffer dstBuffer,VkDeviceSize size)
  {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();
    //copy from src to dst
    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer,srcBuffer,dstBuffer,1,&copyRegion);
    
    endSingleTimeCommands(commandBuffer);
  }
  //find memory type
  uint32_t findMemoryType(uint32_t typeFilter,VkMemoryPropertyFlags properties)
  {
    
    
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice,&memProperties);

    for(uint32_t i= 0;i<memProperties.memoryTypeCount;i++)
      {
	if((typeFilter & (1<<i)) && (memProperties.memoryTypes[i].propertyFlags & properties)==properties)
	  {
	    return i;
	  }
      }
    throw std::runtime_error("Cannot find suitable memory type!");
   
  }//HELPER reate BUFFER fn
  void createBuffer(VkDeviceSize size,VkBufferUsageFlags usage,VkMemoryPropertyFlags properties,VkBuffer& buffer
		    ,VkDeviceMemory& bufferMemory)
  {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage ;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    //bufferInfo.queueFamilyIndexCount =;
    //bufferInfo.pQueueFamilyIndices = ;

    if(vkCreateBuffer(device,&bufferInfo,nullptr,&buffer) != VK_SUCCESS)
      {
	throw std::runtime_error("Vertex buffer not created!");
      }

    //memory requirements
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device,buffer,&memRequirements);

     //memory allocation
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits ,properties);

    if(vkAllocateMemory(device,&allocInfo,nullptr,&bufferMemory) != VK_SUCCESS)
      {
	throw std::runtime_error("Failed to allocate mem to vertex buffer!");
      }
    //bind memory
    vkBindBufferMemory(device,buffer,bufferMemory,0);
  }
  //VERTEX BUFFER
  void createVertexBuffer()
  {
    VkDeviceSize bufferSize = sizeof(vertices[0])*vertices.size();
    //creating a staging buffer with src flag
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    
    createBuffer(bufferSize,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		 stagingBuffer,stagingBufferMemory);
    //filling vertexBuffer
    void* data;
    vkMapMemory(device,stagingBufferMemory,0,bufferSize,0,&data);
    memcpy(data,vertices.data(),(size_t)bufferSize);
    vkUnmapMemory(device,stagingBufferMemory);

    createBuffer(bufferSize,VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		 vertexBuffer,vertexBufferMemory);
    //transfer the data from staging buffer to actual vertex buffer
    copyBuffer(stagingBuffer,vertexBuffer,bufferSize);
    //cleanup staging buffers
    vkDestroyBuffer(device,stagingBuffer,nullptr);
    vkFreeMemory(device,stagingBufferMemory,nullptr);
  }
  void cleanupSwapChain()
  {

    vkDestroyImageView(device, colorImageView, nullptr);
    vkDestroyImage(device, colorImage, nullptr);
    vkFreeMemory(device, colorImageMemory, nullptr);
    
    vkDestroyImageView(device,depthImageView,nullptr);
    vkDestroyImage(device,depthImage,nullptr);
    vkFreeMemory(device,depthImageMemory,nullptr);
    for(auto framebuffer : swapChainFramebuffers)
      {
	vkDestroyFramebuffer(device,framebuffer,nullptr);
      }
    for(auto imageView : swapChainImageViews)
      {
	vkDestroyImageView(device,imageView,nullptr);
      }

    vkDestroySwapchainKHR(device,swapChain,nullptr);
  }
  //RECREATE SWAP CHAIN
  void recreateSwapChain()
  {
    int width =0;
    int height =0;
    glfwGetFramebufferSize(window,&width,&height);
    while(width == 0 || height ==0)
      {
	glfwGetFramebufferSize(window,&width,&height);
	glfwWaitEvents();
      }
    vkDeviceWaitIdle(device);

    cleanupSwapChain();//to cleanup the below objects
    
    createSwapChain();
    createImageViews();
    createColorResources();
    createDepthResources();
    createFramebuffers();
  }
  //create SYNCHRONIZATION USING SEMAPHORES
  void createSyncObjects()
  {
    imageAvailableSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFence.resize(MAX_FRAMES_IN_FLIGHT);
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; 

    for(int i=0;i<MAX_FRAMES_IN_FLIGHT;i++)
      {
	if(vkCreateSemaphore(device,&semaphoreInfo,nullptr,&imageAvailableSemaphore[i])!= VK_SUCCESS ||
	   vkCreateSemaphore(device,&semaphoreInfo,nullptr,&renderFinishedSemaphore[i]) != VK_SUCCESS ||
	   vkCreateFence(device,&fenceInfo,nullptr,&inFlightFence[i]) != VK_SUCCESS)
      {
	throw std::runtime_error("Semaphores and fence could not be created!");
      }
      }
  }
  //CREATE COMMAND BUFFER
  void createCommandBuffer()
  {
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

    if(vkAllocateCommandBuffers(device,&allocInfo,commandBuffers.data()) != VK_SUCCESS)
      {
	throw std::runtime_error("Could not create command buffer!");
      }
  }
  //function that write command into the command buffer
  void recordCommandBuffer(VkCommandBuffer commandBuffer,uint32_t imageIndex)
  {
    //clear values
    std::array<VkClearValue,2> clearValues{};
    clearValues[0].color = {0.0f,0.0f,0.0f,1.0f};
    clearValues[1].depthStencil = {1.0f,0};
    //command buffer  begin
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    beginInfo.pInheritanceInfo =nullptr;

    if(vkBeginCommandBuffer(commandBuffer,&beginInfo) != VK_SUCCESS)
      {
	throw std::runtime_error("Could not create begin command buffer!");
      }
    //starting a render pass
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer =swapChainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset ={0,0};
    renderPassInfo.renderArea.extent = swapChainExtent;
    //VkClearValue clearColor = {{{0.0f,0.0f,0.0f,1.0f}}};
    renderPassInfo.clearValueCount =static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();
    //begin render pass
    vkCmdBeginRenderPass(commandBuffer,&renderPassInfo,VK_SUBPASS_CONTENTS_INLINE);
    

    //set viewport
    VkViewport viewport{};
    viewport.x =0.0f;
    viewport.y =0.0f;
    viewport.width =static_cast<float>(swapChainExtent.width);
    viewport.height =static_cast<float>(swapChainExtent.height);
    viewport.minDepth =0.0f;
    viewport.maxDepth =1.0f;
    vkCmdSetViewport(commandBuffer,0,1,&viewport);
    //set scissor
    VkRect2D scissor{};
    scissor.offset ={0,0};
    scissor.extent =swapChainExtent;
    vkCmdSetScissor(commandBuffer,0,1,&scissor);
    //bind graphics pipeline
    vkCmdBindPipeline(commandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,graphicsPipeline);

    VkBuffer vertexBuffers[] = {vertexBuffer};
    VkDeviceSize offsets[] = {0};

    vkCmdBindVertexBuffers(commandBuffer,0,1,vertexBuffers,offsets);
    //bind index buffer
    vkCmdBindIndexBuffer(commandBuffer,indexBuffer,0,VK_INDEX_TYPE_UINT32);
    //issue draw command
    //vkCmdDraw(commandBuffer,static_cast<uint32_t>(vertices.size()),1,0,0);
    
    //BIND DESCRIPTOR SETS
    vkCmdBindDescriptorSets(commandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,pipelineLayout,0,1,&descriptorSets[currentFrame],0,nullptr);
    //ISSUE INDEXED DRAW CMD
    vkCmdDrawIndexed(commandBuffer,static_cast<uint32_t>(indices.size()),1,0,0,0);

    //end render pass
    vkCmdEndRenderPass(commandBuffer);

    //end command buffer
    if(vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
      {
	throw std::runtime_error("Failed to record command buffer!");
      }
  }
  //CREATE COMMAND POOL
  void createCommandPool()
  {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    if(vkCreateCommandPool(device,&poolInfo,nullptr,&commandPool) != VK_SUCCESS)
      {
	throw std::runtime_error("Could not create CommandPool");
      }
  }
  //FRAMEBUFFER CREATEION
  void createFramebuffers()
  {
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for(size_t i=0;i<swapChainImageViews.size();i++)
      {
	std::array<VkImageView,3> attachments ={
	  colorImageView,
	  depthImageView,
	  swapChainImageViews[i]
	 
       
	};

	VkFramebufferCreateInfo framebufferInfo{};
	framebufferInfo.sType= VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	framebufferInfo.renderPass = renderPass;
	framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
	framebufferInfo.pAttachments = attachments.data();
	framebufferInfo.width = swapChainExtent.width;
	framebufferInfo.height = swapChainExtent.height;
	framebufferInfo.layers =1;
	if(vkCreateFramebuffer(device,&framebufferInfo,nullptr,&swapChainFramebuffers[i]) != VK_SUCCESS)
	  {
	    throw std::runtime_error("Could not create framebuffer !");
	  }
      }
  }
  //create render pass
  void createRenderPass()
  {
    //for depth
    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = msaaSamples;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    
    
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format =swapChainImageFormat;
    colorAttachment.samples = msaaSamples;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp =VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    //color attachment ref
    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    //attachment for color
    VkAttachmentDescription colorAttachmentResolve{};
    colorAttachmentResolve.format = swapChainImageFormat;
    colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    //attachment ref
    VkAttachmentReference colorAttachmentResolveRef{};
    colorAttachmentResolveRef.attachment = 2;
    colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    
    
    //subpass
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount =1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.pResolveAttachments = &colorAttachmentResolveRef;
     //subpass dependencies
    VkSubpassDependency dependency{};
    dependency.srcSubpass =VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass =0;
    dependency.srcStageMask =VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    //render pass
    std::array<VkAttachmentDescription,3> attachments = {colorAttachment,depthAttachment,colorAttachmentResolve};
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount =static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount =1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies =&dependency;

    if(vkCreateRenderPass(device,&renderPassInfo,nullptr,&renderPass) != VK_SUCCESS)
      {
	throw std::runtime_error("Could not create render pass.\n");
      }
    
    
  }
  //create GRAPHICS PIPELINE
  void createGraphicsPipeline()
  {
    
    std::vector<char> vertShaderCode   = readFile("shaders/vert.spv");
    std::vector<char> fragShaderCode = readFile("shaders/frag.spv");
    
    VkShaderModule vertShaderModule= createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
   

    //create pipeline stage
    //vertex shader stage
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};

    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    //vertShaderStageInfo.pNext= ;
    //vertShaderStageInfo.flags=;
    vertShaderStageInfo.stage=VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module=vertShaderModule;
    vertShaderStageInfo.pName ="main";
    //vertShaderStageInfo.pSpecializationInfo=;

    //fragment shader stage
    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};

    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    //arrya of both shader stages
    VkPipelineShaderStageCreateInfo shaderStages[]={ vertShaderStageInfo, fragShaderStageInfo };

    //Vertex Input descriptions
    //VkVertexInputBindingDescription bindingDescription = Vertex::getBindingDescription();
    //std::array<VkVertexInputAttributeDescription,3> attributeDescriptions = Vertex::getAttributeDescriptions();
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();
    //Vertex Input
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount=static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
    //INPUT ASSEMBLY STAGE
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    //VIEWPORT AND SCISSOR
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    //SCISSOR TO DISPLAY ENTIRE EXTENT
    VkRect2D scissor{};
    scissor.offset = {0,0};
    scissor.extent = swapChainExtent;
    //STATIC STATE
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;
    //RASTERZATION
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;
    //MULTISAMPLING (DISABLED FOR NOW)
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_TRUE;
    multisampling.rasterizationSamples = msaaSamples;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;
    //COLOR BLENDING ATTACHMENT STATE
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp=VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    //COLOR BLENDING STATE CREATE INFO
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] =0.0f;
    colorBlending.blendConstants[1] =0.0f;
    colorBlending.blendConstants[2] =0.0f;
    colorBlending.blendConstants[3] =0.0f;
    //dynamics states
    std::vector<VkDynamicState> dynamicStates ={
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates= dynamicStates.data();
    //PIPELINE LAYOUT

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    if(vkCreatePipelineLayout(device,&pipelineLayoutInfo,nullptr,&pipelineLayout)!=VK_SUCCESS)
      {
	throw std::runtime_error("Failed to create Pipeline Layout.\n");
      }
    else
      {
	std::cout<<"GRAPHICS PIPELINE CREATED"<<std::endl;
      }

    //DEPTH AND STENCIL STATE
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f;
    depthStencil.maxDepthBounds = 1.0f;
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {};
    depthStencil.back= {};

    //create PIPELINE
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    //pipelineInfo.flags= ;
    pipelineInfo.stageCount =2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    //pipelineInfo.pTessellationState = ;
    pipelineInfo.pViewportState =&viewportState;
    pipelineInfo.pRasterizationState =&rasterizer;
    pipelineInfo.pMultisampleState =&multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;;
    pipelineInfo.layout= pipelineLayout;
    pipelineInfo.renderPass =renderPass;
    pipelineInfo.subpass= 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    //pipelineInfo.basePipelineIndex = -1;
    //create pipeline
    if(vkCreateGraphicsPipelines(device,VK_NULL_HANDLE,1,&pipelineInfo,nullptr,&graphicsPipeline)!=VK_SUCCESS)
      {
	throw std::runtime_error("Could not create graphics pipeline.\n");
      }
    //destory shader module after use ,shadermodule are thing wrapper around shader byte code
    vkDestroyShaderModule(device,fragShaderModule,nullptr);
    vkDestroyShaderModule(device,vertShaderModule,nullptr);
    
  }
  //create a shader module to transfer the shader code to pipeline
  VkShaderModule createShaderModule(const std::vector<char>& code)
  {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    //createInfo.pNext = nullptr;
    //createInfo.flags = ;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    //create shader module
    VkShaderModule shaderModule;
    if(vkCreateShaderModule(device,&createInfo,nullptr,&shaderModule) != VK_SUCCESS)
      {
	throw std::runtime_error("Shader module could not be created.\n");
      }
    return shaderModule;
  }
  //create IMAGE VIEWS
  void createImageViews()
  {
    swapChainImageViews.resize(swapChainImages.size());
    for(size_t i =0;i<swapChainImages.size();i++)
      {
	swapChainImageViews[i] = createImageView(swapChainImages[i],swapChainImageFormat,VK_IMAGE_ASPECT_COLOR_BIT,1);
      }
    
    
  }
  //create SWAP-CHAIN
  void createSwapChain()
  {
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode     = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent                = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount  = swapChainSupport.capabilities.minImageCount + 1;//min images count + one

    if(swapChainSupport.capabilities.maxImageCount >0 && imageCount > swapChainSupport.capabilities.maxImageCount)//if max count is set to more than 0 and
      //if mind image count is greate than set max image count
      {
	imageCount = swapChainSupport.capabilities.maxImageCount;
      }
    //swap chain object
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount  = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    //swapchain images across multiple queue families
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[]={indices.graphicsFamily.value(),indices.presentFamily.value()};
    if(indices.graphicsFamily != indices.presentFamily)
      {
	createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
	createInfo.queueFamilyIndexCount = 2;
	createInfo.pQueueFamilyIndices = queueFamilyIndices;
      }
    else
      {
	createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	createInfo.queueFamilyIndexCount = 0;
	createInfo.pQueueFamilyIndices = nullptr;
      }
    createInfo.preTransform  = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    //createInfo.oldSwapchain = VK_NULL_HANDLE;

    //create swap chain
    if(vkCreateSwapchainKHR(device,&createInfo,nullptr,&swapChain) != VK_SUCCESS)
      {
	throw std::runtime_error("SWAPCHAIN could not be created!\n");
      }
    //retrieve swap chain images
    vkGetSwapchainImagesKHR(device,swapChain,&imageCount,nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device,swapChain,&imageCount,swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent      = extent;
    
  }
  //create surface
  void createSurface()
  {
    if(glfwCreateWindowSurface(instance,window,nullptr,&surface) != VK_SUCCESS)
      {
	throw std::runtime_error("Failed to create window surface");
      }
  }
  //Creating a logical device
  void createLogicalDevice()
  {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    //create two queue for graphics queue and presentation queue
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),indices.presentFamily.value()};
    float queuePriority = 1.0f;
    for(uint32_t queueFamily : uniqueQueueFamilies)
      {
	VkDeviceQueueCreateInfo queueCreateInfo{};
	queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCreateInfo.queueFamilyIndex = queueFamily;
	queueCreateInfo.queueCount  = 1;
       	queueCreateInfo.pQueuePriorities = &queuePriority;

	//push back the current queue
	queueCreateInfos.push_back(queueCreateInfo);
      }
    
    //device features
    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    deviceFeatures.sampleRateShading = VK_TRUE;

    //create logical device
    VkDeviceCreateInfo  createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());//(update)presentation and render queue
    createInfo.pQueueCreateInfos = queueCreateInfos.data();//updated after presentation and render queue
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidationLayers)
      {
	createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
	createInfo.ppEnabledLayerNames = validationLayers.data();
      }
    else
      {
	createInfo.enabledLayerCount = 0;
      }
    //createdevice
    if(vkCreateDevice(physicalDevice,&createInfo,nullptr,&device) != VK_SUCCESS)
      {
	throw std::runtime_error("FAILED TO CREATE LOGICAL DEVICE.");
      }
    //get device queue
    //vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    //get graphcis and compute queue
    vkGetDeviceQueue(device,indices.graphicsFamily.value(),0, &graphicsQueue);
    //retirieve queue handle
    vkGetDeviceQueue(device,indices.presentFamily.value(),0,&presentQueue);
  }
  
  //pick physical device
  void pickPhysicalDevice()
  {
    //VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    //list graphics card
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance,&deviceCount,nullptr);
    std::cout<<"PHYSICAL DEVICE COUNT : "<<deviceCount<<std::endl;

    if(deviceCount == 0)
      {
	throw std::runtime_error("No physical device present.\n");
      }
    //allocate new handle to store physical address handle
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance,&deviceCount,devices.data());

    for(const VkPhysicalDevice& device: devices)
      {
	if(isDeviceSuitable(device))
	  {
	    physicalDevice = device;
	    msaaSamples = getMaxUsableSampleCountBits();
	    break;
	  }
      }

    if(physicalDevice == VK_NULL_HANDLE)
      {
	throw std::runtime_error("Failed to find a suitable GPU");
      }
  }
  //POPULATE SWAP CHAIN SUPPORT STRUCt
  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
  {
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device,surface,&details.capabilities);//surface query

    //check for surface formats
    uint32_t formatCount;
    
    vkGetPhysicalDeviceSurfaceFormatsKHR(device,surface,&formatCount,nullptr);

    if(formatCount!=0)
      {
	details.formats.resize(formatCount);
	vkGetPhysicalDeviceSurfaceFormatsKHR(device,surface,&formatCount,details.formats.data());
      }
    //check for presentation formats
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device,surface,&presentModeCount,nullptr);

    if(presentModeCount != 0)
      {
	details.presentModes.resize(presentModeCount);
	vkGetPhysicalDeviceSurfacePresentModesKHR(device,surface,&presentModeCount,details.presentModes.data());
      }

    return details;
  }
  //populate debug messenger create info
  void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
  {
    createInfo={};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
  }
  void setupDebugMessenger()
  {
    if(!enableValidationLayers)
      {
	return;
      }
    //fill in createmessengerinfo struct

    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    populateDebugMessengerCreateInfo(createInfo);

    //create vkdebuglUitlsmessengerEXt object
    if(CreateDebugUtilsMessengerEXT(instance,&createInfo,nullptr,&debugMessenger)!=VK_SUCCESS)
      {
	throw std::runtime_error("failed to setup a debug messenger!");
      }
      
  }
  void mainLoop()
  {
    //main window loop
    while(!glfwWindowShouldClose(window))
      {
	glfwPollEvents();
	drawFrame();
      }
    vkDeviceWaitIdle(device);

  }
  //DRAW FRAME FUNC
  void drawFrame()
  {
   
   
    //wait for prev frame to finish
    vkWaitForFences(device,1,&inFlightFence[currentFrame],VK_TRUE,UINT64_MAX);
    //acquiring an image from swap chain
    uint32_t imageIndex;
   

    VkResult result = vkAcquireNextImageKHR(device,swapChain,UINT64_MAX,imageAvailableSemaphore[currentFrame],VK_NULL_HANDLE,&imageIndex);
    
    if(result == VK_ERROR_OUT_OF_DATE_KHR)
      {
	recreateSwapChain();
	return;
      }
    else if( result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
      {
	 throw std::runtime_error("Failed to aquire swap chain images");
      }
    //UPDATE UNIFORM BUFFER
    updateUniformBuffer(currentFrame);
    //reset fences(only reset the fence if we are submitting work)
    vkResetFences(device,1,&inFlightFence[currentFrame]);
    
    //reset command buffer
    vkResetCommandBuffer(commandBuffers[currentFrame],0);
    //now call the created func to create the command buffer for the image index
    recordCommandBuffer(commandBuffers[currentFrame],imageIndex);
    //submitting command buffer
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    //semaphore array
    VkSemaphore waitSemaphores[] = {imageAvailableSemaphore[currentFrame]};
    VkPipelineStageFlags waitStages[]={VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask =waitStages;
    submitInfo.commandBufferCount =1;
    submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
    //signale semaphores
    VkSemaphore signalSemaphores[] = {renderFinishedSemaphore[currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    //vkResetFences(device,1,&inFlightFence[currentFrame]);
    if(vkQueueSubmit(graphicsQueue,1,&submitInfo,inFlightFence[currentFrame]) != VK_SUCCESS)
      {
	throw std::runtime_error("Failed to submit draw command buffer!");
      }
    //presentation
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount =1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    VkSwapchainKHR swapChains[] = {swapChain};
    presentInfo.swapchainCount =1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    //presentInfo.pResults = nullptr;
    //present finally d:1/10/23
    result = vkQueuePresentKHR(presentQueue,&presentInfo);

    if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
       {
	 framebufferResized = false;
	 recreateSwapChain();
	 
       }
     else if( result != VK_SUCCESS)
       {
	 throw std::runtime_error("Failed to aquire swap chain images");
       }
     

    currentFrame = (currentFrame +1)% MAX_FRAMES_IN_FLIGHT;
    
  }
  //UPDATE UNIFORM BUFFER
  void updateUniformBuffer(uint32_t currentImage)
  {
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();

    float time = std::chrono::duration<float,std::chrono::seconds::period>(currentTime - startTime).count();

    //fill the ubo struct
    UniformBufferObject ubo{};
    ubo.model = glm::rotate(glm::mat4(1.0f),time*glm::radians(90.0f),glm::vec3(0.0f,0.0f,1.0f));
    ubo.view = glm::lookAt(glm::vec3(2.0f,2.0f,2.0f),glm::vec3(0.0f,0.0f,0.0f),glm::vec3(0.0f,0.0f,1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f),swapChainExtent.width/(float)swapChainExtent.height
				,0.1f,10.0f);
    ubo.proj[1][1] *= -1; //glm was designed for opengl but in vulkan the y coord is flipped
    //copy data of ubo to current uniform buffer object
    memcpy(uniformBuffersMapped[currentImage],&ubo,sizeof(ubo));
  }
  void cleanup()
  {
    //cleanup swapchain
    cleanupSwapChain();

    vkDestroySampler(device,textureSampler,nullptr);
    vkDestroyImageView(device,textureImageView,nullptr);

    vkDestroyImage(device, textureImage, nullptr);
    vkFreeMemory(device, textureImageMemory, nullptr);
    
    //destroy uniform buffers handle and device memory
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
      {
	vkDestroyBuffer(device,uniformBuffers[i],nullptr);
	vkFreeMemory(device,uniformBuffersMemory[i],nullptr);
      }
    //destroy descriptor pool
    vkDestroyDescriptorPool(device,descriptorPool,nullptr);
    //Cleanup descriptor set layout
    vkDestroyDescriptorSetLayout(device,descriptorSetLayout,nullptr);
    //destroy index buffer
    vkDestroyBuffer(device,indexBuffer,nullptr);
    vkFreeMemory(device,indexBufferMemory,nullptr);
    
    //destroy vertexBuffer
    vkDestroyBuffer(device,vertexBuffer,nullptr);
    vkFreeMemory(device,vertexBufferMemory,nullptr);
    
    //cleanup semaphores and fence
    for(int i=0;i<MAX_FRAMES_IN_FLIGHT;i++)
      {
	vkDestroySemaphore(device,renderFinishedSemaphore[i],nullptr);
	vkDestroySemaphore(device,imageAvailableSemaphore[i],nullptr);
	vkDestroyFence(device,inFlightFence[i],nullptr);
      }
    //destroy command pool
    vkDestroyCommandPool(device,commandPool,nullptr);
    //destroy framebuffer
    /* for(auto framebuffer : swapChainFramebuffers)
      {
	vkDestroyFramebuffer(device,framebuffer,nullptr);
      }
    */
    //destroy pipeline
    vkDestroyPipeline(device,graphicsPipeline,nullptr);
    //destroy pipeline layout
    vkDestroyPipelineLayout(device,pipelineLayout,nullptr);
    //destroy render pass
    vkDestroyRenderPass(device,renderPass,nullptr);
    //destroy image views
    /*for(VkImageView& imageView : swapChainImageViews)
      {
	vkDestroyImageView(device,imageView,nullptr);
      }
    
    //destroy swapchain
    vkDestroySwapchainKHR(device,swapChain,nullptr);
    */
    //destroy commandpool
    //vkDestroyCommandPool(device,commandPool,nullptr);
    //destroy device
    vkDestroyDevice(device,nullptr);
    //destroy vkdebugutilsmessengerEXT object
    if(enableValidationLayers)
      {
	DestroyDebugUtilsMessengerEXT(instance,debugMessenger,nullptr);
      }
    vkDestroySurfaceKHR(instance,surface,nullptr);
    vkDestroyInstance(instance,nullptr);
    glfwDestroyWindow(window);//terminate window and glfw
    glfwTerminate();
    
  }
public:
  void createInstance()
  {
    
    //check validation layers
    if(enableValidationLayers && !checkValidationLayerSupport())
      {
	throw std::runtime_error("VALIDATION LAYERS REQUESTED BUT NOT AVAILABLE.!");
      }
    else
      {
	std::cout<<"\t"<<"VALIDATION LAYERS AVAILABLE"<<std::endl;
      }
    
    VkApplicationInfo appInfo{};//fill in struct
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    //appInfo.pNext = nullptr;
    appInfo.pApplicationName = "Hello triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion =VK_MAKE_VERSION(1,0,0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};//fill in antoher sturuct req for instance
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    //createInfo.pNext = nullptr;
    //createInfo.flags = ;
    createInfo.pApplicationInfo = &appInfo;
    //since vulkan is platform agonistic need extension to inteact with window
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;

    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    
    createInfo.enabledExtensionCount = glfwExtensionCount ;
    createInfo.ppEnabledExtensionNames = glfwExtensions;
    
    //CREATION OF CALLBACKS
    auto callback_extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(callback_extensions.size());
    createInfo.ppEnabledExtensionNames = callback_extensions.data();

    //ENABLE AFTER VALIDATIONS LAYERS WERE SET ACTIVE
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if(enableValidationLayers)
      {
	createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
	createInfo.ppEnabledLayerNames = validationLayers.data();

	populateDebugMessengerCreateInfo(debugCreateInfo);
	createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
      }
    else
      {
	createInfo.enabledLayerCount = 0;
	createInfo.pNext = nullptr;
      }

    //creating instance pointer to creation infor struct, pointer to allocator callback,pointer to variable that stores the handle of new object
    //VkResult result = vkCreateInstance(&createInfo,nullptr,&instance);
    

    if(vkCreateInstance(&createInfo,nullptr,&instance) != VK_SUCCESS)
    {
      throw std::runtime_error("Failed to create an instance!\n");
    }
    //checking for extension support
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr,&extensionCount,nullptr);//stores no of extensions and an array of VkExtensionProperties to store details of extensions
    //allocate array to hold extensions details
    std::vector<VkExtensionProperties> extensions(extensionCount);
    //finally query extension details
    vkEnumerateInstanceExtensionProperties(nullptr,&extensionCount,extensions.data());

    //each VkExtensionProperties contain name and version of extension can use them
    std::cout<<"Available extensions: \n";
    for(const  VkExtensionProperties& extension : extensions)
      {
	std::cout<<"\t"<<extension.extensionName <<"\n";
      }

    //check if all required extensions are present
    printf("LIST OF EXTENSIONS REQUIRED BY GLFW:\n");
    for(auto i=0;i<glfwExtensionCount;i++)
      {
	std::cout<<"\t"<<glfwExtensions[i]<<std::endl;
      }
    checkRequiredExtension(extensions,glfwExtensions,glfwExtensionCount);
  }
  //CHECK VALIDATION LAYER AVAILABILITY
  bool checkValidationLayerSupport()
  {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount,nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount,availableLayers.data());

    for(const char* layerName: validationLayers)
      {
	bool layerFound = false;
	for(const VkLayerProperties& layerProperties : availableLayers)
	  {
	    if(strcmp(layerName,layerProperties.layerName) == 0)
	      {
		layerFound = true;
		break;
	      }
	    
	    
	  }
	if(!layerFound)
	  {
	    return false;
	  }
      }

    return true;
    
  }

  //STATIC DEBUG CALLBACK FUNCTION
  static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback( VkDebugUtilsMessageSeverityFlagBitsEXT  messageSeverity,
						       VkDebugUtilsMessageTypeFlagsEXT messageType,
						       const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
						       void* pUserData)
  {
    std::cerr<<"VALIDATION LAYER :"<<pCallbackData->pMessage <<std::endl;

    return VK_FALSE;
  }
  //CHECK IF PHYSICAL DEVICE IS SUITABLE
  bool isDeviceSuitable(VkPhysicalDevice device)
  {
    QueueFamilyIndices indices = findQueueFamilies(device);

    //d:23/9/23
    //function to check availability of swap chain
    bool extensionsSupported = checkDeviceExtensionsSupport(device);
    //VERIFY SWAP CHAIN SUPPORT IS ADEAQUATE
    bool swapChainAdequate = false;
    if(extensionsSupported)
      {
	SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
	swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
      }
    //d:20/9/23
    //adding physical device properties and features
    /* VkPhysicalDeviceProperties deviceProperties;
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceProperties(device,&deviceProperties);
    vkGetPhysicalDeviceFeatures(device,&deviceFeatures);
    */
    VkPhysicalDeviceFeatures supportedFeatures;
    vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
    
    return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
  }

  //func to checkdevice extensions(swap chain)
  bool checkDeviceExtensionsSupport(VkPhysicalDevice device)
  {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device,nullptr,&extensionCount,nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device,nullptr,&extensionCount,availableExtensions.data());
    std::set<std::string>requiredExtensions(deviceExtensions.begin(),deviceExtensions.end());

    for(const VkExtensionProperties& extension : availableExtensions)
      {
	requiredExtensions.erase(extension.extensionName);
      }
    return requiredExtensions.empty();
    
    
    return true;
  }
  //Find queue family
  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
  {
    QueueFamilyIndices indices;
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device,&queueFamilyCount,nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device,&queueFamilyCount,queueFamilies.data());
    //check if display to surface is supported by present family queue
    VkBool32 presentSupport = false;
    //find one queue family that supports  VK_QUEUE_GRAPHICS_BIT.
    int i =0 ;
    for(const VkQueueFamilyProperties& queueFamily : queueFamilies)
      {
	vkGetPhysicalDeviceSurfaceSupportKHR(device,i,surface,&presentSupport);
	//update below for computer shader
	if((queueFamily.queueFlags &  VK_QUEUE_GRAPHICS_BIT))
	  {
	    indices.graphicsFamily = i;
	  }
	if(presentSupport)
	  {
	    indices.presentFamily = i;
	  }
	
	if(indices.isComplete())
	  {
	    std::cout<<" VK_QUEUE_GRAPHICS_BIT present in GPU"<<std::endl;
	    break;
	  }
	i++;
      }
    

    return indices;
  }
  //SURFACE FORMAT(SWAP CHAIN): colro depth
  VkSurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<VkSurfaceFormatKHR> &availableFormats)
  {
    for(const VkSurfaceFormatKHR& availableFormat : availableFormats)
      {
	if(availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
	  {
	    return availableFormat;
	  }
      }
    return availableFormats[0];
  }
  //PRESENT MODE SELECTION
  VkPresentModeKHR chooseSwapPresentMode(std::vector<VkPresentModeKHR>&availablePresentModes)
  {
    for(const VkPresentModeKHR& availablePresentMode : availablePresentModes)
      {
	if(availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
	  {
	    return availablePresentMode;
	  }
      }
    return VK_PRESENT_MODE_FIFO_KHR;
  }

  //SWAP EXTENT(SCREEN RESOLUTION OF VULKAN SURFACE)
  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
  {
    if(capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
      {
	return capabilities.currentExtent;
      }
    else
      {
	int width,height;
	glfwGetFramebufferSize(window,&width,&height);
	VkExtent2D actualExtent = {
	  static_cast<uint32_t>(width),
	  static_cast<uint32_t>(height)
	};
	actualExtent.width = std::clamp(actualExtent.width,capabilities.minImageExtent.width,capabilities.maxImageExtent.width);
	actualExtent.height = std::clamp(actualExtent.height,capabilities.minImageExtent.height,capabilities.maxImageExtent.height);

	return actualExtent;
      }
    
  }
  //Heper function to load in data
  static std::vector<char> readFile(const std::string& filename)
  {
    std::ifstream file(filename,std::ios::ate | std::ios::binary);
    if(!file.is_open())
      {
	throw std::runtime_error("File could not be created.\n");
      }
    //read from end of the file and set the buffer with file side
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    //seek beginning and read all bytes at once
    file.seekg(0);
    file.read(buffer.data(),fileSize);
    file.close();

    return buffer;
  }
private:
  GLFWwindow* window;//window

  VkInstance instance;//instance
  VkDebugUtilsMessengerEXT debugMessenger;
  VkSurfaceKHR surface;//window surface
  
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device;
  
  VkQueue graphicsQueue;
  VkQueue presentQueue;//for window surface -> this is inside that logical device, presentation queue and normal queue are created
  VkQueue computeQueue;
  
  VkSwapchainKHR swapChain;//swapchain handle
  std::vector<VkImage>swapChainImages;//retrieve images
  VkFormat swapChainImageFormat;
  VkExtent2D swapChainExtent;
  std::vector<VkImageView>swapChainImageViews;//create image views
  std::vector<VkFramebuffer> swapChainFramebuffers;//fraim buffers
  
  VkRenderPass renderPass;//render pass
  VkDescriptorSetLayout descriptorSetLayout;//descriptor set layout handle
  VkPipelineLayout pipelineLayout;//unifrom stuff later on
  VkPipeline graphicsPipeline;//the true "pipeline"
  
  VkCommandPool commandPool;//create command pool

  //MIPMAP
  uint32_t mipLevels;
  //Texture image
  VkImage textureImage;
  VkDeviceMemory textureImageMemory;
  VkImageView textureImageView;//images are accessed through image views rather than directly
  VkSampler textureSampler;

  //VkBuffer vertexBuffer;//vertex buffer handle
  //VkDeviceMemory vertexBufferMemory;//allocated vertex buffer memory
  VkBuffer indexBuffer;//index buffer
  VkDeviceMemory indexBufferMemory;

  //unfirom buffer stuff
  std::vector<VkBuffer> uniformBuffers;
  std::vector<VkDeviceMemory>uniformBuffersMemory;
  std::vector<void*>uniformBuffersMapped;

  //descriptorpool handle
  VkDescriptorPool descriptorPool;
  std::vector<VkDescriptorSet>descriptorSets;
  
  // VkCommandBuffer commandBuffer;// command buffer handle
  std::vector<VkCommandBuffer> commandBuffers;//concurrent cmdbuffer
  
  //semaphores and fence handle
  /*VkSemaphore imageAvailableSemaphore;
  VkSemaphore renderFinishedSemaphore;
  VkFence inFlightFence;*/
  std::vector<VkSemaphore> imageAvailableSemaphore;
  std::vector<VkSemaphore> renderFinishedSemaphore;
  std::vector<VkFence> inFlightFence;
  uint32_t currentFrame = 0;

  bool framebufferResized = false;

  //for depth buffer
  VkImage depthImage;
  VkDeviceMemory depthImageMemory;
  VkImageView depthImageView;

  //for model
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  VkBuffer vertexBuffer;
  VkDeviceMemory vertexBufferMemory;

  //sample count
  VkSampleCountFlagBits msaaSamples  = VK_SAMPLE_COUNT_1_BIT;
  VkImage colorImage;
  VkDeviceMemory colorImageMemory;
  VkImageView colorImageView;

  //for compute shader
  std::vector<VkBuffer> shaderStorageBuffers;
  std::vector<VkDeviceMemory> shaderStorageBuffersMemory;
 

};
int main()
{
  HelloTriangleApplication app;
  try
    {
      app.run();
    }
  catch(const std::exception& e)
    {
      std::cerr<<e.what()<<std::endl;
      return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}
