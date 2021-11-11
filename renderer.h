// minimalistic code to draw a single triangle, this is not part of the API.
// TODO: Part 1b
#include "shaderc/shaderc.h" // needed for compiling shaders at runtime
#ifdef _WIN32 // must use MT platform DLL libraries on windows
	#pragma comment(lib, "shaderc_combined.lib") 
#endif
#include "build/FSLogo.h"
#include "h2bParser.h"
#include <stdlib.h>  
#include <fstream>
#include <string>


#define DEGREES_TO_RADIANS(fDegrees) (fDegrees * 3.14159 / 180.0f)
#define RADIANS_TO_DEGREES(fRadians) (fRadians / 3.14159 * 180.0f)
// Simple Vertex Shader
#define MAX_SUBMESH_PER_DRAW 1024
const char* vertexShaderSource = R"(
// TODO: 2i
// an ultra simple hlsl vertex shader
// TODO: Part 2b

//[[vk::push_constant]]
//cbuffer MESH_INDEX
//{
//uint mesh_ID;
//};

#pragma pack_matrix(row_major)
struct OBJ_ATTRIBUTES
{
float3    Kd; 
float	    d;  
float3     Ks; 
float       Ns; 
float3     Ka; 
float       sharpness; 
float3     Tf; 
float       Ni; 
float3     Ke; 
uint   illum; 
};
struct SHADER_MODEL_DATA
	{
		float4 sunDirection,sunColor;
		matrix viewMatrix, ProjectionMatrix;
		matrix matrices[1024];
		OBJ_ATTRIBUTES materials[1024];
		float4 ambient,camPos;
	};
StructuredBuffer<SHADER_MODEL_DATA> SceneData;
// TODO: Part 4g
// TODO: Part 2i
// TODO: Part 3e
// TODO: Part 4a
[[vk::push_constant]]
cbuffer MESH_INDEX
{
uint mesh_ID;
};

struct OUTPUT_TO_RASTER
{
float4 posH : SV_POSITION;
float3 nrmW : NORMAL;
float3 posW : WORLD;
};
// TODO: Part 1f
struct Vertex
{
	float3 xyz : POSITION;
	float3 uvm : TEXCORD;
	float3 nrm : NORMAL;
};
// TODO: Part 4b
OUTPUT_TO_RASTER main(Vertex inputVertex) : SV_POSITION
{
    // TODO: Part 1h
float4 pos = float4(inputVertex.xyz,1);
pos = mul(pos,SceneData[0].matrices[mesh_ID]);
pos = mul(pos,SceneData[0].viewMatrix);
pos = mul(pos,SceneData[0].ProjectionMatrix);
	
	// TODO: Part 2i
		// TODO: Part 4e
	// TODO: Part 4b
OUTPUT_TO_RASTER output;
output.posH = pos;
output.nrmW = mul(normalize(inputVertex.nrm),SceneData[0].matrices[mesh_ID]);
output.posW = mul(inputVertex.xyz,(float3x3)SceneData[0].matrices[mesh_ID]);
return output;
		// TODO: Part 4e
}
)";
// Simple Pixel Shader
const char* pixelShaderSource = R"(
// TODO: Part 2b
// TODO: Part 4g
// TODO: Part 2i
// TODO: Part 3e
struct OUTPUT_TO_RASTER
{
float4 posH : SV_POSITION;
float3 nrmW : NORMAL;
float3 posW : WORLD;
};

[[vk::push_constant]]
cbuffer MESH_INDEX
{
uint mesh_ID;
};
// an ultra simple hlsl pixel shader
// TODO: Part 4b
#pragma pack_matrix(row_major)
struct OBJ_ATTRIBUTES
{
float3    Kd; 
float	    d;  
float3     Ks; 
float       Ns; 
float3     Ka; 
float       sharpness; 
float3     Tf; 
float       Ni; 
float3     Ke; 
uint   illum; 
};
struct SHADER_MODEL_DATA
	{
		float4 sunDirection,sunColor;
		matrix viewMatrix, ProjectionMatrix;
		matrix matrices[1024];
		OBJ_ATTRIBUTES materials[1024];
		float4 ambient,camPos;
	};
StructuredBuffer<SHADER_MODEL_DATA> SceneData;
float4 main(OUTPUT_TO_RASTER output) : SV_TARGET 
{	
	float lr = saturate(dot(-SceneData[0].sunDirection,output.nrmW));
	float alr = saturate(SceneData[0].ambient + lr);
	
	float3 result = alr * SceneData[0].materials[mesh_ID].Kd * SceneData[0].sunColor;
	float3 vd = normalize(SceneData[0].camPos - output.posW);
	float3 hv = normalize((-SceneData[0].sunDirection) + vd);
	float3 intensity = max(pow(saturate(dot(output.nrmW,hv)),SceneData[0].materials[mesh_ID].Ns),0);
	float3 rl = SceneData[0].sunColor * intensity * SceneData[0].materials[mesh_ID].Ks;
	return float4(result + rl,1);
	// TODO: Part 1a
	// TODO: Part 3a
	// TODO: Part 4c
	// TODO: Part 4g (half-vector or reflect method your choice)
}
)";
// Creation, Rendering & Cleanup
class Renderer
{
	struct Vertex
	{
		float xyz[3];
		float uvm[3];
		float nrm[3];
	};
	// TODO: Part 2b

	struct SHADER_MODEL_DATA
	{
		GW::MATH::GVECTORF sunDirection, sunColor;
		GW::MATH::GMATRIXF viewMatrix, ProjectionMatrix;
		GW::MATH::GMATRIXF matrices[MAX_SUBMESH_PER_DRAW];
		OBJ_ATTRIBUTES materials[MAX_SUBMESH_PER_DRAW];
		GW::MATH::GVECTORF ambient, camPos;
	};
	SHADER_MODEL_DATA smd;
	struct model
	{
		GW::MATH::GMATRIXF world;
		std::string name = "";
	};
	std::vector<model> models;
	// proxy handles
	GW::SYSTEM::GWindow win;
	GW::GRAPHICS::GVulkanSurface vlk;
	GW::CORE::GEventReceiver shutdown;

	// what we need at a minimum to draw a triangle
	VkDevice device = nullptr;
	VkBuffer vertexHandle = nullptr;
	VkDeviceMemory vertexData = nullptr;
	// TODO: Part 1g

	unsigned int maxFrames;

	//Creating Keyboard mouse proxy
	GW::INPUT::GInput inp;

	VkBuffer indexHandle = nullptr;
	VkDeviceMemory indexData = nullptr;
	// TODO: Part 2c
	std::vector<VkBuffer> storagebuffer;
	std::vector<VkDeviceMemory> storagedata;
	VkShaderModule vertexShader = nullptr;
	VkShaderModule pixelShader = nullptr;
	// pipeline settings for drawing (also required)
	VkPipeline pipeline = nullptr;
	VkPipelineLayout pipelineLayout = nullptr;
	// TODO: Part 2e
	VkDescriptorSetLayout layout = nullptr;
	// TODO: Part 2f
	VkDescriptorPool dpool = nullptr;
	// TODO: Part 2g
	std::vector<VkDescriptorSet> set;
	// TODO: Part 4f
	std::chrono::steady_clock::time_point end, start;
	// TODO: Part 2a
	// TODO: Part 2b
	struct MESH_INDEX
	{
		unsigned mesh_ID;
	};
	MESH_INDEX mi;
	// TODO: Part 4g
	GW::MATH::GMATRIXF world;
	GW::MATH::GMatrix gmat;
	GW::MATH::GMATRIXF view ;
	GW::MATH::GMATRIXF projection ;
	GW::MATH::GVector vec;
	GW::MATH::GMATRIXF rotation;
	GW::MATH::GMATRIXF wtemp;
	Parser r;
public:

	Renderer(GW::SYSTEM::GWindow _win, GW::GRAPHICS::GVulkanSurface _vlk)
	{
		inp.Create(_win);
		gmat.Create();
		vec.Create();
		win = _win;
		vlk = _vlk;
		unsigned int width, height;
		win.GetClientWidth(width);
		win.GetClientHeight(height);
		// TODO: Part 2a
		// TODO: Part 2b
		// TODO: Part 4g
		// TODO: part 3b
		gmat.IdentityF(world);
		gmat.IdentityF(rotation);
		GW::MATH::GVECTORF eye = {0.75f,0.25f,-1.5f,0};
		GW::MATH::GVECTORF at = {0.15f,0.75f,0,0};
		GW::MATH::GVECTORF up = { 0,1,0,0 };
		gmat.LookAtLHF(eye, at, up, view);
		float fov = DEGREES_TO_RADIANS(65);
		float aspect;
		vlk.GetAspectRatio(aspect);
		float np = 0.1f;
		float fp = 100.0f;
		gmat.ProjectionVulkanLHF(fov, aspect, np, fp, projection);	
		GW::MATH::GVECTORF ldir = { -1 , -1 , 2 };
		vec.NormalizeF(ldir,ldir);
		GW::MATH::GVECTORF lcolor = {0.9f,0.9f,1.0f,1.0f}; 

		/***************** GEOMETRY INTIALIZATION ******************/
		// Grab the device & physical device so we can allocate some stuff
		
		VkPhysicalDevice physicalDevice = nullptr;
		vlk.GetDevice((void**)&device);
		vlk.GetPhysicalDevice((void**)&physicalDevice);
		smd.ProjectionMatrix = projection;
		smd.viewMatrix = view;
		smd.sunColor = lcolor;
		smd.sunDirection = ldir;
		smd.ambient = {0.25f , 0.25f,0.35f,0};
		smd.camPos = eye;
		for (int i = 0; i < FSLogo_materialcount; i++)
		{
			smd.materials[i] = FSLogo_materials[i].attrib;
		}
		smd.matrices[0] = world;
		smd.matrices[1] = rotation;
		// TODO: Part 1c
		// Create Vertex Buffer
		// Transfer triangle data to the vertex buffer. (staging would be prefered here)
		GvkHelper::create_buffer(physicalDevice, device, sizeof(FSLogo_vertices),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &vertexHandle, &vertexData);
		GvkHelper::write_to_buffer(device, vertexData, FSLogo_vertices, sizeof(FSLogo_vertices));
		// TODO: Part 1g
		GvkHelper::create_buffer(physicalDevice, device, sizeof(FSLogo_indices),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &indexHandle, &indexData);
		GvkHelper::write_to_buffer(device, indexData, FSLogo_indices, sizeof(FSLogo_indices));
		// TODO: Part 2d
		
		vlk.GetSwapchainImageCount(maxFrames);
		storagebuffer.resize(maxFrames);
		storagedata.resize(maxFrames);
		for (int i = 0; i < maxFrames; i++)
		{
			GvkHelper::create_buffer(physicalDevice, device, sizeof(smd),
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &storagebuffer[i], &storagedata[i]);
			GvkHelper::write_to_buffer(device, storagedata[i], &smd, sizeof(smd));
		}
		/***************** SHADER INTIALIZATION ******************/
		// Intialize runtime shader compiler HLSL -> SPIRV
		shaderc_compiler_t compiler = shaderc_compiler_initialize();
		shaderc_compile_options_t options = shaderc_compile_options_initialize();
		shaderc_compile_options_set_source_language(options, shaderc_source_language_hlsl);
		shaderc_compile_options_set_invert_y(options, false); // TODO: Part 2i
#ifndef NDEBUG
		shaderc_compile_options_set_generate_debug_info(options);
#endif
		// Create Vertex Shader
		shaderc_compilation_result_t result = shaderc_compile_into_spv( // compile
			compiler, vertexShaderSource, strlen(vertexShaderSource),
			shaderc_vertex_shader, "main.vert", "main", options);
		if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) // errors?
			std::cout << "Vertex Shader Errors: " << shaderc_result_get_error_message(result) << std::endl;
		GvkHelper::create_shader_module(device, shaderc_result_get_length(result), // load into Vulkan
			(char*)shaderc_result_get_bytes(result), &vertexShader);
		shaderc_result_release(result); // done
		// Create Pixel Shader
		result = shaderc_compile_into_spv( // compile
			compiler, pixelShaderSource, strlen(pixelShaderSource),
			shaderc_fragment_shader, "main.frag", "main", options);
		if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) // errors?
			std::cout << "Pixel Shader Errors: " << shaderc_result_get_error_message(result) << std::endl;
		GvkHelper::create_shader_module(device, shaderc_result_get_length(result), // load into Vulkan
			(char*)shaderc_result_get_bytes(result), &pixelShader);
		shaderc_result_release(result); // done
		// Free runtime shader compiler resources
		shaderc_compile_options_release(options);
		shaderc_compiler_release(compiler);

		/***************** PIPELINE INTIALIZATION ******************/
		// Create Pipeline & Layout (Thanks Tiny!)
		VkRenderPass renderPass;
		vlk.GetRenderPass((void**)&renderPass);
		VkPipelineShaderStageCreateInfo stage_create_info[2] = {};
		// Create Stage Info for Vertex Shader
		stage_create_info[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage_create_info[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stage_create_info[0].module = vertexShader;
		stage_create_info[0].pName = "main";
		// Create Stage Info for Fragment Shader
		stage_create_info[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage_create_info[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stage_create_info[1].module = pixelShader;
		stage_create_info[1].pName = "main";
		// Assembly State
		VkPipelineInputAssemblyStateCreateInfo assembly_create_info = {};
		assembly_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assembly_create_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assembly_create_info.primitiveRestartEnable = false;
		// TODO: Part 1e
		// Vertex Input State
		VkVertexInputBindingDescription vertex_binding_description = {};
		vertex_binding_description.binding = 0;
		vertex_binding_description.stride = sizeof(float) * 9;
		vertex_binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		VkVertexInputAttributeDescription vertex_attribute_description[3] = {
			{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 },
			{ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12},
			{ 2, 0, VK_FORMAT_R32G32B32_SFLOAT, 24}
			//uv, normal, etc....
		};
		VkPipelineVertexInputStateCreateInfo input_vertex_info = {};
		input_vertex_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		input_vertex_info.vertexBindingDescriptionCount = 1;
		input_vertex_info.pVertexBindingDescriptions = &vertex_binding_description;
		input_vertex_info.vertexAttributeDescriptionCount = 3;
		input_vertex_info.pVertexAttributeDescriptions = vertex_attribute_description;
		// Viewport State (we still need to set this up even though we will overwrite the values)
		VkViewport viewport = {
            0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1
        };
        VkRect2D scissor = { {0, 0}, {width, height} };
		VkPipelineViewportStateCreateInfo viewport_create_info = {};
		viewport_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewport_create_info.viewportCount = 1;
		viewport_create_info.pViewports = &viewport;
		viewport_create_info.scissorCount = 1;
		viewport_create_info.pScissors = &scissor;
		// Rasterizer State
		VkPipelineRasterizationStateCreateInfo rasterization_create_info = {};
		rasterization_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterization_create_info.rasterizerDiscardEnable = VK_FALSE;
		rasterization_create_info.polygonMode = VK_POLYGON_MODE_FILL;
		rasterization_create_info.lineWidth = 1.0f;
		rasterization_create_info.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterization_create_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterization_create_info.depthClampEnable = VK_FALSE;
		rasterization_create_info.depthBiasEnable = VK_FALSE;
		rasterization_create_info.depthBiasClamp = 0.0f;
		rasterization_create_info.depthBiasConstantFactor = 0.0f;
		rasterization_create_info.depthBiasSlopeFactor = 0.0f;
		// Multisampling State
		VkPipelineMultisampleStateCreateInfo multisample_create_info = {};
		multisample_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisample_create_info.sampleShadingEnable = VK_FALSE;
		multisample_create_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisample_create_info.minSampleShading = 1.0f;
		multisample_create_info.pSampleMask = VK_NULL_HANDLE;
		multisample_create_info.alphaToCoverageEnable = VK_FALSE;
		multisample_create_info.alphaToOneEnable = VK_FALSE;
		// Depth-Stencil State
		VkPipelineDepthStencilStateCreateInfo depth_stencil_create_info = {};
		depth_stencil_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depth_stencil_create_info.depthTestEnable = VK_TRUE;
		depth_stencil_create_info.depthWriteEnable = VK_TRUE;
		depth_stencil_create_info.depthCompareOp = VK_COMPARE_OP_LESS;
		depth_stencil_create_info.depthBoundsTestEnable = VK_FALSE;
		depth_stencil_create_info.minDepthBounds = 0.0f;
		depth_stencil_create_info.maxDepthBounds = 1.0f;
		depth_stencil_create_info.stencilTestEnable = VK_FALSE;
		// Color Blending Attachment & State
		VkPipelineColorBlendAttachmentState color_blend_attachment_state = {};
		color_blend_attachment_state.colorWriteMask = 0xF;
		color_blend_attachment_state.blendEnable = VK_FALSE;
		color_blend_attachment_state.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_COLOR;
		color_blend_attachment_state.dstColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR;
		color_blend_attachment_state.colorBlendOp = VK_BLEND_OP_ADD;
		color_blend_attachment_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		color_blend_attachment_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;
		color_blend_attachment_state.alphaBlendOp = VK_BLEND_OP_ADD;
		VkPipelineColorBlendStateCreateInfo color_blend_create_info = {};
		color_blend_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		color_blend_create_info.logicOpEnable = VK_FALSE;
		color_blend_create_info.logicOp = VK_LOGIC_OP_COPY;
		color_blend_create_info.attachmentCount = 1;
		color_blend_create_info.pAttachments = &color_blend_attachment_state;
		color_blend_create_info.blendConstants[0] = 0.0f;
		color_blend_create_info.blendConstants[1] = 0.0f;
		color_blend_create_info.blendConstants[2] = 0.0f;
		color_blend_create_info.blendConstants[3] = 0.0f;
		// Dynamic State 
		VkDynamicState dynamic_state[2] = { 
			// By setting these we do not need to re-create the pipeline on Resize
			VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamic_create_info = {};
		dynamic_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamic_create_info.dynamicStateCount = 2;
		dynamic_create_info.pDynamicStates = dynamic_state;
		
		// TODO: Part 2e
		VkDescriptorSetLayoutBinding layoutBinding = {};
		layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
		layoutBinding.descriptorCount = 1;
		layoutBinding.binding = 0;
		layoutBinding.pImmutableSamplers = nullptr;
		VkDescriptorSetLayoutCreateInfo layoutCreateInfo = {};
		layoutCreateInfo.bindingCount = 1;
		layoutCreateInfo.pBindings = &layoutBinding;
		layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutCreateInfo.flags = 0;
		layoutCreateInfo.pNext = nullptr;
		vkCreateDescriptorSetLayout(device, &layoutCreateInfo, nullptr, &layout);
		
		

		

		
		// TODO: Part 2f
		
		VkDescriptorPoolSize poolsize;
		poolsize.descriptorCount = maxFrames;
		poolsize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		VkDescriptorPoolCreateInfo poolCreateInfro = {};
		poolCreateInfro.flags = 0;
		poolCreateInfro.pPoolSizes = &poolsize;
		poolCreateInfro.maxSets = maxFrames;
		poolCreateInfro.poolSizeCount = 1;
		poolCreateInfro.pNext = nullptr;
		poolCreateInfro.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		vkCreateDescriptorPool(device, &poolCreateInfro, nullptr, &dpool);
		
			// TODO: Part 4f
		// TODO: Part 2g
		VkDescriptorSetAllocateInfo allocateInfo = {};
		allocateInfo.descriptorPool = dpool;
		allocateInfo.descriptorSetCount = 1;
		allocateInfo.pSetLayouts = &layout;
		allocateInfo.pNext = nullptr;
		allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		set.resize(maxFrames);
		for (int i = 0; i < maxFrames; i++)
		{
			VkResult result =  vkAllocateDescriptorSets(device, &allocateInfo, &set[i]);
		}
		// TODO: Part 4f	
		// TODO: Part 2h
		
		
		VkWriteDescriptorSet writeDescriptorSet = {};
		writeDescriptorSet.descriptorCount = 1;
		writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writeDescriptorSet.dstBinding = 0;
		writeDescriptorSet.dstArrayElement = 0;
		for (int  i = 0; i < storagedata.size(); i++)
		{
			writeDescriptorSet.dstSet = set[i];
			VkDescriptorBufferInfo bufferInfo = { storagebuffer[i],0,VK_WHOLE_SIZE };
			writeDescriptorSet.pBufferInfo = &bufferInfo;
			vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
		}
		
		
		
		// TODO: Part 4f
	
		// Descriptor pipeline layout
		VkPushConstantRange constantRange = {};
		constantRange.offset = 0;
		constantRange.size = 128;
		constantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
		VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
		pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		// TODO: Part 2e
		pipeline_layout_create_info.setLayoutCount = 1;
		pipeline_layout_create_info.pSetLayouts = &layout;
		// TODO: Part 3c
		pipeline_layout_create_info.pushConstantRangeCount = 1;
		pipeline_layout_create_info.pPushConstantRanges = &constantRange;
		vkCreatePipelineLayout(device, &pipeline_layout_create_info, 
			nullptr, &pipelineLayout);
	    // Pipeline State... (FINALLY) 
		VkGraphicsPipelineCreateInfo pipeline_create_info = {};
		pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeline_create_info.stageCount = 2;
		pipeline_create_info.pStages = stage_create_info;
		pipeline_create_info.pInputAssemblyState = &assembly_create_info;
		pipeline_create_info.pVertexInputState = &input_vertex_info;
		pipeline_create_info.pViewportState = &viewport_create_info;
		pipeline_create_info.pRasterizationState = &rasterization_create_info;
		pipeline_create_info.pMultisampleState = &multisample_create_info;
		pipeline_create_info.pDepthStencilState = &depth_stencil_create_info;
		pipeline_create_info.pColorBlendState = &color_blend_create_info;
		pipeline_create_info.pDynamicState = &dynamic_create_info;
		pipeline_create_info.layout = pipelineLayout;
		pipeline_create_info.renderPass = renderPass;
		pipeline_create_info.subpass = 0;
		pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
		vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, 
			&pipeline_create_info, nullptr, &pipeline);

		/***************** CLEANUP / SHUTDOWN ******************/
		// GVulkanSurface will inform us when to release any allocated resources
		shutdown.Create(vlk, [&]() {
			if (+shutdown.Find(GW::GRAPHICS::GVulkanSurface::Events::RELEASE_RESOURCES, true)) {
				CleanUp(); // unlike D3D we must be careful about destroy timing
			}
		});
	}
	void Render()
	{
		end = std::chrono::steady_clock::now();
		std::chrono::duration<double> diff = start - end;
		gmat.RotateYLocalF(rotation, diff.count(), rotation);
		start = end;
		smd.matrices[0] = world;
		smd.matrices[1] = rotation;
		// TODO: Part 2a
		// TODO: Part 4d
		// grab the current Vulkan commandBuffer
		unsigned int currentBuffer;
		vlk.GetSwapchainCurrentImage(currentBuffer);
		VkCommandBuffer commandBuffer;
		vlk.GetCommandBuffer(currentBuffer, (void**)&commandBuffer);
		// what is the current client area dimensions?
		unsigned int width, height;
		win.GetClientWidth(width);
		win.GetClientHeight(height);
		// setup the pipeline's dynamic settings
		VkViewport viewport = {
            0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1
        };
        VkRect2D scissor = { {0, 0}, {width, height} };
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		
		// now we can draw
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexHandle, offsets);
		//vkCmdDraw(commandBuffer, 3885, 1, 0, 0);
		// TODO: Part 1h
		vkCmdBindIndexBuffer(commandBuffer, indexHandle, 0, VK_INDEX_TYPE_UINT32);
		GvkHelper::write_to_buffer(device, storagedata[currentBuffer], &smd, sizeof(smd));
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &set[currentBuffer], 0, 0);
		for (int i = 0; i < FSLogo_meshcount; i++)
		{
			
			vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(unsigned), &FSLogo_meshes[i].materialIndex);
			vkCmdDrawIndexed(commandBuffer, FSLogo_meshes[i].indexCount, 1, FSLogo_meshes[i].indexOffset, 0, 0);
		}
	}
	void UpdateCamera()
	{	gmat.InverseF(smd.viewMatrix, wtemp);
		auto now = std::chrono::steady_clock::now();
		float dur = std::chrono::duration_cast<std::chrono::microseconds>(now - end).count() / 1000000.0f;
		end = now;
		const float CameraSpeed = 0.3f;
		float ychange = 0;
		float conxchange = 0, conzchange = 0;
		GW::MATH::GMATRIXF trans = GW::MATH::GIdentityMatrixF;
		GW::MATH::GVECTORF translation = { 0,0,0 };
		if (G_PASS(inp.GetState(G_KEY_SPACE, ychange)) && ychange) {
			translation.y = ychange * dur * CameraSpeed;
			gmat.TranslateGlobalF(trans, translation, trans);
			gmat.MultiplyMatrixF(trans, wtemp, wtemp);
		}
		else if (G_PASS(inp.GetState(G_KEY_LEFTSHIFT, ychange)) && ychange) {
			translation.y = -ychange * dur * CameraSpeed;
			gmat.TranslateGlobalF(trans, translation, trans);
			gmat.MultiplyMatrixF(trans, wtemp, wtemp);
		}
		if(G_PASS(inp.GetState(G_KEY_A, ychange)) && ychange) {
			translation.x = -ychange * dur * CameraSpeed;
			gmat.TranslateGlobalF(trans, translation, trans);
			gmat.MultiplyMatrixF(trans, wtemp, wtemp);
		}
		else if (G_PASS(inp.GetState(G_KEY_D, ychange)) && ychange)  {
			translation.x = ychange * dur * CameraSpeed;
			gmat.TranslateGlobalF(trans, translation, trans);
			gmat.MultiplyMatrixF(trans, wtemp, wtemp);
		}
		if (G_PASS(inp.GetState(G_KEY_W, ychange)) && ychange) {
			translation.z = ychange * dur * CameraSpeed ;
			gmat.TranslateGlobalF(trans, translation, trans);
			gmat.MultiplyMatrixF(trans, wtemp, wtemp);
		}
		else if (G_PASS(inp.GetState(G_KEY_S, ychange)) && ychange)  {
			translation.z = -ychange * dur * CameraSpeed ;
			gmat.TranslateGlobalF(trans, translation, trans);
			gmat.MultiplyMatrixF(trans, wtemp, wtemp);
		}
		float x, y, total, aspect;
		unsigned int hei, wid;
		float thumbspeed = 3.142857142 * dur;
		GW::GReturn result = inp.GetMouseDelta(x, y);
		float fov = DEGREES_TO_RADIANS(65);
		win.GetHeight(hei);
		win.GetWidth(wid);
		vlk.GetAspectRatio(aspect);
		GW::MATH::GVECTORF save = { 0,0,0,1 };
		float conyresult = 0;
		float conxresult = 0;
		

		if ((G_PASS(result) && result != GW::GReturn::REDUNDANT))
		{
			conyresult *= -thumbspeed;
			total = ((fov * y) / hei) + (conyresult);
			gmat.RotateXLocalF(wtemp, total, wtemp);
		}
		if (G_PASS(result) && result != GW::GReturn::REDUNDANT)
		{
			conxresult *= thumbspeed;
			total = ((fov * aspect * x) / wid) + conxresult;
			save = wtemp.row4;
			gmat.RotateYGlobalF(wtemp, total, wtemp);
			wtemp.row4 = save;
		}
		gmat.InverseF(wtemp, smd.viewMatrix);
		
	}
	void textfile(const char* file)
	{
		model curr;
		std::string s;
		std::ifstream ifl;
		ifl.open(file);
		if (ifl.is_open() == false)
		{
			return ;
		}
		char buffer[256];
		std::string temp = "";
		while (!ifl.eof())
		{
			std::getline(ifl, s);
			if (s == "MESH")
			{
				temp = "";
				std::getline(ifl, s);
				for (size_t i = 0; i < s.size(); i++)
				{
					if (s[i] == '.' || s[i] == ' ')
					{
						break;
					}
					temp += s[i];
				}
				curr.name = temp;
				ifl.getline(buffer, 256, '(');
				ifl.getline(buffer, 256, ',');
				curr.world.row1.x = std::stof(buffer);
				ifl.getline(buffer, 256, ',');
				curr.world.row1.y = std::stof(buffer);
				ifl.getline(buffer, 256, ',');
				curr.world.row1.z = std::stof(buffer);
				ifl.getline(buffer, 256, ')');
				curr.world.row1.w = std::stof(buffer);

				ifl.getline(buffer, 256, '(');
				ifl.getline(buffer, 256, ',');
				curr.world.row2.x = std::stof(buffer);
				ifl.getline(buffer, 256, ',');
				curr.world.row2.y = std::stof(buffer);
				ifl.getline(buffer, 256, ',');
				curr.world.row2.z = std::stof(buffer);
				ifl.getline(buffer, 256, ')');
				curr.world.row2.w = std::stof(buffer);

				ifl.getline(buffer, 256, '(');
				ifl.getline(buffer, 256, ',');
				curr.world.row3.x = std::stof(buffer);
				ifl.getline(buffer, 256, ',');
				curr.world.row3.y = std::stof(buffer);
				ifl.getline(buffer, 256, ',');
				curr.world.row3.z = std::stof(buffer);
				ifl.getline(buffer, 256, ')');
				curr.world.row3.w = std::stof(buffer);

				ifl.getline(buffer, 256, '(');
				ifl.getline(buffer, 256, ',');
				curr.world.row4.x = std::stof(buffer);
				ifl.getline(buffer, 256, ',');
				curr.world.row4.y = std::stof(buffer);
				ifl.getline(buffer, 256, ',');
				curr.world.row4.z = std::stof(buffer);
				ifl.getline(buffer, 256, ')');
				curr.world.row4.w = std::stof(buffer);
				models.push_back(curr);
			}
		}
		ifl.close();
	
	}
	
private:
	void CleanUp()
	{
		// wait till everything has completed
		vkDeviceWaitIdle(device);
		// Release allocated buffers, shaders & pipeline
		// TODO: Part 1g
		vkDestroyBuffer(device, indexHandle, nullptr);
		vkFreeMemory(device, indexData, nullptr);
		// TODO: Part 2d
		vkDestroyBuffer(device, vertexHandle, nullptr);
		vkFreeMemory(device, vertexData, nullptr);
		vkDestroyShaderModule(device, vertexShader, nullptr);
		vkDestroyShaderModule(device, pixelShader, nullptr);
		// TODO: Part 2e
		vkDestroyDescriptorSetLayout(device, layout, nullptr);
		// TODO: part 2f
		vkDestroyDescriptorPool(device, dpool, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyPipeline(device, pipeline, nullptr);

		for (int i = 0; i < maxFrames; i++)
		{
			vkDestroyBuffer(device, storagebuffer[i], nullptr);
			vkFreeMemory(device, storagedata[i], nullptr);
		}

	}
};
