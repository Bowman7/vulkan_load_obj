VULKAN_SDK_PATH = /usr/include/vulkan
STB_INCLUDE_PATH = /usr/local/include/stb/
TINYOBJ_INCLUDE_PATH = /usr/local/include/tinyobjloader/

CFLAGS = -std=c++17 -I$(VULKAN_SDK_PATH)/include -I$(STB_INCLUDE_PATH) -I$(TINYOBJ_INCLUDE_PATH)

LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

#compile

VulkanTest: main.cpp
	g++ $(CFLAGS) -o VulkanTest main.cpp $(LDFLAGS) -O3

#phony prevent confusion with file name

.PHONY: test clean

test: VulkanTest
	./VulkanTest
clean:
	rm -f VulkanTest
