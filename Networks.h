#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <limits.h>
#include <ctime>
#include <stdlib.h>
#include <stdio.h>
#include <list>
#include <cmath>

#include "Layers.h"

class Network
{
public:
	layer_base** layers;
	int layer_count;
	int input_size;
	int output_size;
	float_t learning_rate;
	int adaptive;
	float_t adaptive_rate;
	float_t prev_error;
	Network()
	{
		layers = 0;
		layer_count = 0;
		input_size = 0;
		output_size = 0;
		adaptive = 0;
		prev_error = -1.0f;
		adaptive_rate = 1.0f;
		learning_rate = 0.1f;
	};

	~Network()
	{
		for (int i = 0; i < layer_count; i++)
		{
			delete layers[i];
		}
		delete[] layers;
	}

	void test()
	{
		layers = new layer_base*[2];
		layers[0] = new layer_fully_connected(2, 3);
		layers[1] = new layer_fully_connected(3, 1);
		layers[0]->randomize();
		layers[1]->randomize();

		float_t** ddset_in = new float_t*[4];
		float_t** setout = new float_t*[4];
	
		for (int i = 0; i < 4; i++)
		{
			ddset_in[i]=new float_t[2];
			setout[i] = new float_t[1];
		}

		ddset_in[0][0] = 0.0;
		ddset_in[0][1] = 0.0;
		ddset_in[1][0] = 0.0;
		ddset_in[1][1] = 1.0;
		ddset_in[2][0] = 1.0;
		ddset_in[2][1] = 0.0;
		ddset_in[3][0] = 1.0;
		ddset_in[3][1] = 1.0;

		setout[0][0] = 0.0;
		setout[1][0] = 1.0;
		setout[2][0] = 1.0;
		setout[3][0] = 0.0;
		int count = 0;
		float_t error = 1.0;
		int threshold_correct = 0;
		int tcount = 0;
		while (threshold_correct<4)
		{
			tcount++;
			count++;
			error = 0.0;
			threshold_correct = 0;
			float_t* gradient_initial;

			layers[0]->reset_weights();
			layers[1]->reset_weights();

			for (int i = 0; i < 4; i++)
			{
				float_t* rt1 = layers[0]->forward_pass(ddset_in[i]);
				
				float_t * tmp= layers[1]->forward_pass(rt1);

				error += (tmp[0] - setout[i][0])*(tmp[0] - setout[i][0]);

				

				if ( abs( layers[1]->output[0] - setout[i][0] )<=0.01f)
					threshold_correct++;
				
				gradient_initial = layers[1]->calculate_initial_gradient(setout[i]);

				layers[0]->back_pass(layers[1]->back_pass(gradient_initial, learning_rate), learning_rate);
				


				if (threshold_correct >= 4)
				{
					std::cout << std::endl;
					for (int i = 0; i < 4; i++)
					{
						float_t* rt1 = layers[0]->forward_pass(ddset_in[i]);
						float_t * tmp = layers[1]->forward_pass(rt1);
						std::cout << "In1 " << ddset_in[i][0] << "   In2 " << ddset_in[i][1] <<
							"   out " << tmp[0] << std::endl;
					}
				}

				layers[0]->update_weights();
				layers[1]->update_weights();

			}

			
			
			if (count > 1000)
			{
				std::cout << "out " << threshold_correct << "   ";
				std::cout << error << std::endl;
				count = 0;
			}
		}
		
		std::cout << "Final error ";
		std::cout << error << std::endl;
		error = 0.0;
		std::cout <<  "total count " << tcount << std::endl;




	}

	void randomize()
	{
		for (int ln = 0; ln < layer_count; ln++)
		{
			layers[ln]->randomize();
		}
	};

	void set_learning_rate(float_t _learning_rate)
	{
		learning_rate = _learning_rate;
	}

	void add_layer(layer_base* to_add)
	{
		if (layer_count == 0)
		{
			layers = new layer_base*[1];
			layers[0] = to_add;
			layer_count++;
			input_size = layers[0]->num_input;
			output_size = layers[0]->num_output;
			return;
		}

		if (layers[layer_count - 1]->num_output != to_add->num_input)
		{
			perror("incompatible input/output size\n");
			return;
		}

		layer_base** tmp = new layer_base*[layer_count + 1];
		for (int i = 0; i < layer_count; i++)
		{
			tmp[i] = layers[i];
		}
		tmp[layer_count] = to_add;
		output_size = to_add->num_output;
		layer_count++;
		delete[] layers;
		layers = tmp;
	}

	float_t calculate_error(float_t** dataset_input, float_t** dataset_output, int dataset_size)
	{
		float_t error = 0.0;
		for (int i = 0; i < dataset_size; i++)
		{
			float_t* tmp = layers[0]->forward_pass(dataset_input[i]);
			for (int ln = 1; ln < layer_count; ln++)
			{
				tmp = layers[ln]->forward_pass(tmp);
			}

			for (int out = 0; out < output_size; out++)
			{
				error += (tmp[out] - dataset_output[i][out])*(tmp[out] - dataset_output[i][out]);
			}
		}
		return error;
	}

	float_t train(float_t minimum_error, int max_iterations, float_t** dataset_input, float_t** dataset_output, int dataset_size)
	{
		float_t error = calculate_error(dataset_input, dataset_output, dataset_size);
		if (adaptive == 1)
		{
			prev_error = error;
		}

		int counter = 0;
		while ((error > minimum_error) && (max_iterations > counter))
		{
			counter++;
			error = 0.0;

			float_t* gradient_initial;

			for (int ln = 0; ln < layer_count; ln++)
			{
				layers[ln]->reset_weights();
			}

			for (int i = 0; i < dataset_size; i++)
			{


				float_t* tmp = layers[0]->forward_pass(dataset_input[i]);
				for (int ln = 1; ln < layer_count; ln++)
				{
					tmp = layers[ln]->forward_pass(tmp);
				}

				/*for (int out = 0; out < output_size; out++)
				{
					error += abs(tmp[out] - dataset_output[i][out]);
				}*/

				gradient_initial = layers[layer_count - 1]->calculate_initial_gradient(dataset_output[i]);

				for (int ln = (layer_count - 1); ln >= 0; ln--)
				{
					gradient_initial = layers[ln]->back_pass(gradient_initial, learning_rate);


				}
			}
			
			for (int ln = 0; ln < layer_count; ln++)
			{
				layers[ln]->update_weights();
			}

			error = calculate_error(dataset_input, dataset_output, dataset_size);

			if (((prev_error - error) < 0.0) && adaptive == 1)
			{
				learning_rate *= adaptive_rate;
				//printf("%.6f  %.6f %.6f\n", prev_error, error, (prev_error - error));
				//printf("Learning rate %f\n", learning_rate);
			}

		}
		return error;
	};

	int correct(float_t margin, float_t** dataset_input, float_t** dataset_output, int dataset_size)
	{
		int count=0;

		for (int i = 0; i < dataset_size; i++)
		{
			float_t* tmp = layers[0]->forward_pass(dataset_input[i]);
			for (int ln = 1; ln < layer_count; ln++)
			{
				tmp = layers[ln]->forward_pass(tmp);
			}
			int flag = 1;

			for (int out = 0; out < output_size; out++)
			{
				if (abs(tmp[out] - dataset_output[i][out]) > margin)
				{
					flag = 0;
				}
			}

			count += flag;
		}

		return count;
	};

	//Returns a copy of the prediction
	//Dont forget to free
	float_t* predict(float_t* input)
	{
		float_t* tmp = new float_t[output_size];

		float_t* returned = layers[0]->forward_pass(input);
		for (int ln = 1; ln < layer_count; ln++)
		{
			returned = layers[ln]->forward_pass(returned);
		}

		for (int out = 0; out < output_size; out++)
		{
			tmp[out] = returned[out];
		}

		return tmp;
	}

	void set_adaptative(int _adaptive, float_t _rate)
	{
		adaptive = _adaptive;
		adaptive_rate = _rate;
	}


	void save_network(char* file_name)
	{
		std::ofstream ofs;
		ofs.open(file_name, std::ofstream::trunc);
		ofs << "Layer count " << layer_count << std::endl;
		ofs << "Input count " << input_size << std::endl;
		ofs << "Output count " << output_size << std::endl;
		ofs << "Lerning Rate " << learning_rate << std::endl;
		ofs << "Adaptive " << adaptive << std::endl;
		ofs << "Adaptive rate " << adaptive_rate << std::endl;
		ofs << std::endl;

		for (int i = 0; i < layer_count; i++)
		{
			ofs << "Type " << layers[i]->type << std::endl;
			ofs << "Inputs count " << layers[i]->num_input << std::endl;
			ofs << "Output count " << layers[i]->num_output << std::endl;
			ofs << "Weight count " << layers[i]->num_weights << std::endl;
			ofs << "Use momentum " << layers[i]->use_momentum << std::endl;
			ofs << "Decay rate " << layers[i]->decay_rate << std::endl;
			ofs << std::endl;
			ofs << "Weights" << std::endl;
			for (int out = 0; out < (layers[i]->num_output); out++)
			{
				for (int in = 0; in < (layers[i]->num_input); in++)
				{
					ofs << (layers[i]->weights[out][in]) << " ";
				}
				ofs << std::endl;
			}

			if ((layers[i]->use_momentum) == 1)
			{
				ofs << std::endl <<  "Momentum" << std::endl;
				for (int out = 0; out < (layers[i]->num_output); out++)
				{
					for (int in = 0; in < (layers[i]->num_input); in++)
					{
						ofs << (layers[i]->momentum[out][in]) << " ";
					}
					ofs << std::endl;
				}
			}

			ofs << std::endl;
		}

		/*layer_base** layers;
		int layer_count;
		int input_size;
		int output_size;
		float_t learning_rate;
		int adaptive;
		float_t adaptive_rate;
		float_t prev_error;*/
		ofs.close();
	}

	void load_network(char* file_name)
	{
		std::ifstream ifs;
		ifs.open(file_name);

		if (layers != nullptr)
		{
			perror("Network alredy instantated\n");
			return;
		}

		std::string str; //used to discart data
		
		ifs >> str; //Layer
		ifs >> str; //count
		ifs >> layer_count;
		ifs >> str; //Input
		ifs >> str; //count
		ifs >> input_size;
		ifs >> str; //Output
		ifs >> str; //count
		ifs >> output_size;
		ifs >> str; //Learning
		ifs >> str; //Rate
		ifs >> learning_rate;
		ifs >> str; //Adaptative
		ifs >> adaptive;
		ifs >> str; //Adaptative
		ifs >> str; //rate
		ifs >> adaptive_rate;

		layers = new layer_base*[layer_count];

		for (int i = 0; i < layer_count; i++)
		{
			int temp_type;
			int temp_inputs;
			int temp_outputs;
			int temp_weights;
			int temp_use_momentum;
			float_t temp_decay_rate;

			ifs >> str; //Type
			ifs >> temp_type;
			ifs >> str; // Input
			ifs >> str; //Count
			ifs >> temp_inputs;
			ifs >> str; //Output
			ifs >> str; //count
			ifs >> temp_outputs;
			ifs >> str; //Weight
			ifs >> str; //count
			ifs >> temp_weights;
			ifs >> str; // Use
			ifs >> str; //Momentum
			ifs >> temp_use_momentum;
			ifs >> str; //Decay
			ifs >> str; //rate
			ifs >> temp_decay_rate;

			if (temp_type == 0)
			{
				layers[i] = new layer_fully_connected(temp_inputs, temp_outputs);
			}

			if (temp_type == 1)
			{
				layers[i] = new layer_fully_connected_threshold(temp_inputs, temp_outputs);
			}

			if (temp_use_momentum == 1)
			{
				layers[i]->set_momentum(temp_decay_rate);
			}

			ifs >> str; //Weights

			for (int out = 0; out < (layers[i]->num_output); out++)
			{
				for (int in = 0; in < (layers[i]->num_input); in++)
				{
					ifs >> (layers[i]->weights[out][in]);
				}
			}

			if ((layers[i]->use_momentum) == 1)
			{
				ifs >> str; //Momentum
				for (int out = 0; out < (layers[i]->num_output); out++)
				{
					for (int in = 0; in < (layers[i]->num_input); in++)
					{
						ifs >> (layers[i]->momentum[out][in]);
					}
				}
			}
		}

		ifs.close();
	}

};
