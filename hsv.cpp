#include <ogx/Plugins/EasyPlugin.h>
#include <ogx/Data/Clouds/CloudHelpers.h>
#include <ogx/Data/Clouds/KNNSearchKernel.h>
#include <ogx/Data/Clouds/SphericalSearchKernel.h>

using namespace ogx;
using namespace ogx::Data;

struct RGB_to_HSV_conversion : public ogx::Plugin::EasyMethod
{
	// fields
	Nodes::ITransTreeNode* m_node;

	// parameters
	Data::ResourceID m_node_id;
	Real sphere_radius;
	Integer clusters_H;
	Integer clusters_S;
	Integer clusters_V;

	// constructor
	RGB_to_HSV_conversion() : EasyMethod(L"Jaroslaw Affek", L"Average point color in neighborhood. Conversion RGB to HSV.")
	{
	}

	// add input/output parameters
	virtual void DefineParameters(ParameterBank& bank)
	{
		bank.Add(L"node_id", m_node_id).AsNode();
		bank.Add(L"Radius", sphere_radius = 0.6).Min(0.1).Max(3);
		bank.Add(L"Number of H clusters", clusters_H = 4).Min(1).Max(20);
		bank.Add(L"Number of S clusters", clusters_S = 4).Min(1).Max(20);
		bank.Add(L"Number of V clusters", clusters_V = 4).Min(1).Max(20);
	}
	bool Init(Execution::Context& context)
	{
		OGX_SCOPE(log);
		// get node from id
		m_node = context.m_project->TransTreeFindNode(m_node_id);
		if (!m_node) ReportError(L"You must define node_id");

		OGX_LINE.Msg(User, L"Initialization succeeded");
		return EasyMethod::Init(context);
	}

	virtual void Run(Context& context)
	{
		auto subtree = context.Project().TransTreeFindNode(m_node_id);
		// report error if give node was not found, this will stop execution of algorithm
		if (!subtree) ReportError(L"Node not found");

		// update progress every 10000 points
		auto const progress_step = 10000;

		// run with number of threads available on current machine, optional
		auto const thread_count = std::thread::hardware_concurrency();

		// perform calculations for each cloud in given subtree
		Clouds::ForEachCloud(*subtree, [&](Clouds::ICloud & cloud, Nodes::ITransTreeNode & node)
		{
			// access points in the cloud
			Clouds::PointsRange points_all;
			cloud.GetAccess().GetAllPoints(points_all);

			// create vectors for segmentation and feature layers
			std::vector<StoredReal> segmentation_H;
			std::vector<StoredReal> segmentation_S;
			std::vector<StoredReal> segmentation_V;
			std::vector<StoredReal> hue_values;
			std::vector<StoredReal> saturation_values;
			std::vector<StoredReal> value_values;

			Real max_H = -1;
			Real max_S = -1;
			Real max_V = -1;
			Real min_H = 500;
			Real min_S = 500;
			Real min_V = 500;

			auto xyz_r = Data::Clouds::RangeLocalXYZ(points_all);
			auto xyz = xyz_r.begin();
			auto progress = 0;

			for (; xyz != xyz_r.end(); ++xyz)
			{
				// search for neighbors inside sphere
				Clouds::PointsRange neighbors;
				cloud.GetAccess().FindPoints(Clouds::SphericalSearchKernel(Math::Sphere3D(sphere_radius, xyz->cast<Real>())), neighbors);
				auto neighbors_sum = neighbors.size();
				Clouds::RangeColor neighbors_RGB(neighbors);

				// calculate sum of channel values for points in neighborhood
				Real R_sum = 0;
				Real G_sum = 0;
				Real B_sum = 0;
				for (auto & tested_point : neighbors_RGB)
				{
					R_sum += tested_point.x();
					G_sum += tested_point.y();
					B_sum += tested_point.z();
				}
				// calculate average color in neighborhood in RGB values
				Real R_avg = R_sum / neighbors_sum / 255;
				Real G_avg = G_sum / neighbors_sum / 255;
				Real B_avg = B_sum / neighbors_sum / 255;
				
				// RGB to HSV algorithm
				Real max_RGB_value = std::max({R_avg,G_avg,B_avg});
				Real min_RGB_value = std::min({R_avg,G_avg,B_avg});
				Real diff_max_min = max_RGB_value - min_RGB_value;
				// hue
				Real hue = -1;
				if (max_RGB_value == 0 && min_RGB_value == 0)
					hue = 0;
				if (max_RGB_value == R_avg)
					hue = 60 * ((G_avg - B_avg) / diff_max_min);
				if (max_RGB_value == G_avg)
					hue = 60 * (((B_avg - R_avg) / diff_max_min) + 2);
				if (max_RGB_value == B_avg)
					hue = 60 * (((R_avg - G_avg) / diff_max_min) + 4);
				if (hue < 0)
					hue += 360;
				if (hue > max_H)
					max_H = hue;
				if (hue < min_H)
					min_H = hue;
				// saturation
				Real saturation = -1;
				if (max_RGB_value == 0)
					saturation = 0;
				else
					saturation = (diff_max_min / max_RGB_value) * 100;
				if (saturation > max_S)
					max_S = saturation;
				if (saturation < min_S)
					min_S = saturation;
				// value
				Real value = max_RGB_value * 100;
				if (value > max_V)
					max_V = value;
				if (value < min_V)
					min_V = value;
				// save HSV values for every point
				hue_values.push_back(hue);
				saturation_values.push_back(saturation);
				value_values.push_back(value);

				// udpate progress every 10k points and check if we should continue
				++progress;
				if (!(progress % progress_step))
				{
					// progress is from 0 to 1
					if (!context.Feedback().Update(float(progress) / points_all.size()))
					{
						throw EasyException();
					}
				}
			}
			//********************************************************************************************
			// segmentation
			// HUE
			Real cluster_H_val = 0;
			for (auto& iterator : hue_values)
			{
				cluster_H_val = std::floor(float(iterator) / (max_H - min_H) * (clusters_H));
				segmentation_H.push_back(StoredReal(cluster_H_val));
			}
			// create segmentation layer 
			auto v_sLayerName_seg_H = L"segmentation_H";
			Data::Layers::ILayer *layer_seg_H;
			auto layers_seg_H = cloud.FindLayers(v_sLayerName_seg_H);
			// check if layer exist
			if (!layers_seg_H.empty())
				layer_seg_H = layers_seg_H[0];
			else
				layer_seg_H = cloud.CreateLayer(v_sLayerName_seg_H, 0);
			points_all.SetLayerVals(segmentation_H, *layer_seg_H); // saving layer to cloud
			// SATURATION
			Real cluster_S_val = 0;
			for (auto& iterator : saturation_values)
			{
				cluster_S_val = std::floor(float(iterator) / (max_S - min_S) * (clusters_S));
				segmentation_S.push_back(StoredReal(cluster_S_val));
			}
			// create segmentation layer 
			auto v_sLayerName_seg_S = L"segmentation_S";
			Data::Layers::ILayer *layer_seg_S;
			auto layers_seg_S = cloud.FindLayers(v_sLayerName_seg_S);
			// check if layer exist
			if (!layers_seg_S.empty())
				layer_seg_S = layers_seg_S[0];
			else
				layer_seg_S = cloud.CreateLayer(v_sLayerName_seg_S, 0);
			points_all.SetLayerVals(segmentation_S, *layer_seg_S); // saving layer to cloud
			// VALUE
			Real cluster_V_val = 0;
			for (auto& iterator : value_values)
			{
				cluster_V_val = std::floor(float(iterator) / (max_V - min_V) * (clusters_V));
				segmentation_V.push_back(StoredReal(cluster_V_val));
			}
			// create segmentation layer 
			auto v_sLayerName_seg_V = L"segmentation_V";
			Data::Layers::ILayer *layer_seg_V;
			auto layers_seg_V = cloud.FindLayers(v_sLayerName_seg_V);
			// check if layer exist
			if (!layers_seg_V.empty())
				layer_seg_V = layers_seg_V[0];
			else
				layer_seg_V = cloud.CreateLayer(v_sLayerName_seg_V, 0);
			points_all.SetLayerVals(segmentation_V, *layer_seg_V); // saving layer to cloud

			//**********************************************************************************
			// create feature layers
			//HUE
			auto v_sLayerName_H = L"HUE";
			Data::Layers::ILayer *layer_H;
			auto layers_H = cloud.FindLayers(v_sLayerName_H);
			// check if layer exist
			if (!layers_H.empty())
				layer_H = layers_H[0];
			else
				layer_H = cloud.CreateLayer(v_sLayerName_H, 0);
			points_all.SetLayerVals(hue_values, *layer_H); // saving layer to cloud
			//SATURATION
			auto v_sLayerName_S = L"SATURATION";
			Data::Layers::ILayer *layer_S;
			auto layers_S = cloud.FindLayers(v_sLayerName_S);
			// check if layer exist
			if (!layers_S.empty())
				layer_S = layers_S[0];
			else
				layer_S = cloud.CreateLayer(v_sLayerName_S, 0);
			points_all.SetLayerVals(saturation_values, *layer_S); // saving layer to cloud
			//VALUE
			auto v_sLayerName_V = L"VALUE";
			Data::Layers::ILayer *layer_V;
			auto layers_V = cloud.FindLayers(v_sLayerName_V);
			// check if layer exist
			if (!layers_V.empty())
				layer_V = layers_V[0];
			else
				layer_V = cloud.CreateLayer(v_sLayerName_V, 0);
			points_all.SetLayerVals(value_values, *layer_V); // saving layer to cloud


		}, thread_count); // run with given number of threads, optional parameter, if not given will run in current thread
	}
};

OGX_EXPORT_METHOD(RGB_to_HSV_conversion)