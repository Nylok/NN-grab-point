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


class layer_base
{
public:
	float_t* inputs;
	float_t* output;

	float_t** weights;
	float_t** weights_delta;


	float_t* gradient;
	float_t* tmp;

	int num_input;
	int num_output;
	int num_weights;
	int type;
	
	float_t** momentum;
	int use_momentum;
	float_t decay_rate;

	layer_base()
	{

	}

	virtual ~layer_base()
	{

	}

	virtual void randomize()
	{

	}

	void inject_input(float_t* _inputs, int size)
	{
		if (size != num_input)
		{
			perror("invalid input size");
			return;
		}

		for (int in = 0; in < num_input; in++)
		{
			inputs[in] = in[_inputs];
		}
	};

	virtual void reset_weights()
	{

	};

	virtual void update_weights()
	{

	};

	virtual float_t* forward_pass(float_t* in)
	{
		return nullptr;
	};

	virtual float_t* back_pass(float_t* expected_output, float_t learning_rate)
	{
		return nullptr;
	};

	float_t* calculate_initial_gradient(float* expected_output)
	{
		for (int out = 0; out < num_output; out++)
		{
			tmp[out] = (expected_output[out] - output[out]);
		}
		return tmp;
	};

	virtual void set_momentum(float_t _decay_rate)
	{

	}

};

class layer_fully_connected : public layer_base
{
public:

	layer_fully_connected()
	{

	}
	
	~layer_fully_connected()
	{
		delete[] this->inputs;
		delete[] this->output;
		
		for (int i = 0; i < num_output; i++)
		{
			delete[] this->weights[i];
			delete[] this->weights_delta[i];
		}
		delete[] this->weights;
		delete[] this->weights_delta;
		delete[] this->tmp;
		delete[] this->gradient;
	};

	layer_fully_connected(int _input, int _output)
	{
		this->type = 0;
		this->use_momentum = 0;
		this->num_input = _input;
		this->num_output = _output;
		this->num_weights = _input * _output;

		this->inputs = new float_t[num_input];
		this->output = new float_t[num_output];

		this->weights = new float_t*[num_output];
		this->weights_delta = new float_t*[num_output];
		for (int i = 0; i < num_output; i++)
		{
			this->weights[i] = new float_t[num_input];
			this->weights_delta[i] = new float_t[num_input];
		}
		
		this->gradient = new float_t[num_input];
		this->tmp = new float_t[num_output];
	};

	void randomize()
	{
		for (int out = 0; out < num_output; out++)
		{
			for (int in = 0; in < num_input; in++)
			{
				weights[out][in] = (((float)rand() / (float)RAND_MAX)*2.0f) - 1.0f;
			}
		}
	};

	void reset_weights()
	{
		for (int out = 0; out < num_output; out++)
		{
			for (int in = 0; in < num_input; in++)
			{
				weights_delta[out][in] = 0.0f;
			}
		}

		for (int in = 0; in < num_input; in++)
		{
			gradient[in] = 0.0f;
		}

	};

	void update_weights()
	{
		if (use_momentum == 0)
		{
			for (int out = 0; out < num_output; out++)
			{
				for (int in = 0; in < num_input; in++)
				{
					weights[out][in] += weights_delta[out][in];
				}
			}
		}
		else
		{
			for (int out = 0; out < num_output; out++)
			{
				for (int in = 0; in < num_input; in++)
				{
					momentum[out][in]*=decay_rate;
					weights[out][in] += weights_delta[out][in] + momentum[out][in];
					momentum[out][in] += weights_delta[out][in];
				}
			}
		}
	};

	inline float_t activation_function(float_t val)
	{
		///return log(1.0f + exp(val));
		return 1.0 / (1.0 + exp(-val));
	};

	inline float_t activation_derivate(float_t val)
	{
		///return 1.0 / (1.0 + exp(-val));
		float_t b = activation_function(val);
		return b*(1.0 - b);
	};

	float_t* forward_pass(float_t* in)
	{
		inject_input(in, num_input);

		for (int out = 0; out < num_output; out++)
		{
			output[out] = 0.0f;

			for (int in = 0; in < num_input; in++)
			{
				
				
				output[out] += inputs[in] * weights[out][in];

			}
			output[out] = activation_function(output[out]);
		}

		return output;
	};

	float_t* back_pass(float_t* backprop, float_t learning_rate)
	{
		

		for (int out = 0; out < num_output; out++)
		{
			tmp[out] = activation_derivate(output[out])*backprop[out];
		}

		for (int out = 0; out < num_output; out++)
		{

			for (int in = 0; in < num_input; in++)
			{
				//should never happen
				/*if (isnormal(weights_delta[out][in]) == 0 && output[out] != 0.0)
					printf("damm\n");*/
				weights_delta[out][in] += learning_rate*(tmp[out] * inputs[in]);
			}

		}

		for (int in = 0; in < num_input; in++)
		{
			gradient[in] = 0.0f;
			for (int out = 0; out < num_output; out++)
			{
				gradient[in] += tmp[out] * weights[out][in];
			}
			///std::cout << gradient[in] << " ";
		}
		///std::cout << std::endl;
		///printf("dfgdrfgdsfgsdzfg\n");
		return gradient;

	};

	void set_momentum(float_t _decay_rate)
	{
		this->use_momentum = 1;
		this->decay_rate = _decay_rate;

		this->momentum = new float_t*[num_output];
		for (int i = 0; i < num_output; i++)
		{
			this->momentum[i] = new float_t[num_input];

			for (int j = 0; j < num_input; j++)
			{
				this->momentum[i][j] = 0.0;
			}
		}
	}

};


class layer_fully_connected_threshold : public layer_base
{
public:

	layer_fully_connected_threshold()
	{

	}

	~layer_fully_connected_threshold()
	{
		delete[] this->inputs;
		delete[] this->output;

		for (int i = 0; i < num_output; i++)
		{
			delete[] this->weights[i];
			delete[] this->weights_delta[i];
		}
		delete[] this->weights;
		delete[] this->weights_delta;
		delete[] this->tmp;
		delete[] this->gradient;
	};

	layer_fully_connected_threshold(int _input, int _output)
	{
		this->type = 1;
		this->use_momentum = 0;

		this->num_input = _input;
		this->num_output = _output;
		this->num_weights = _input * _output;

		this->inputs = new float_t[num_input];
		this->output = new float_t[num_output];

		this->weights = new float_t*[num_output];
		this->weights_delta = new float_t*[num_output];
		for (int i = 0; i < num_output; i++)
		{
			this->weights[i] = new float_t[num_input];
			this->weights_delta[i] = new float_t[num_input];
		}

		this->gradient = new float_t[num_input];
		this->tmp = new float_t[num_output];
	};

	void randomize()
	{
		for (int out = 0; out < num_output; out++)
		{
			for (int in = 0; in < num_input; in++)
			{
				weights[out][in] = (((float)rand() / (float)RAND_MAX)*2.0f) - 1.0f;
			}
		}
	};
	
	void reset_weights()
	{
		for (int out = 0; out < num_output; out++)
		{
			for (int in = 0; in < num_input; in++)
			{
				weights_delta[out][in] = 0.0f;
			}
		}

		for (int in = 0; in < num_input; in++)
		{
			gradient[in] = 0.0f;
		}

	};

	void update_weights()
	{
		if (use_momentum == 0)
		{
			for (int out = 0; out < num_output; out++)
			{
				for (int in = 0; in < num_input; in++)
				{
					weights[out][in] += weights_delta[out][in];
				}
			}
		}
		else
		{
			for (int out = 0; out < num_output; out++)
			{
				for (int in = 0; in < num_input; in++)
				{
					momentum[out][in] *= decay_rate;
					weights[out][in] += weights_delta[out][in] + momentum[out][in];
					momentum[out][in] += weights_delta[out][in];
				}
			}
		}
	};

	inline float_t activation_function(float_t val)
	{
		return 1.0 / (1.0 + exp(-10.0*val));
	};

	inline float_t activation_derivate(float_t val)
	{
		float_t p = exp(10.0*val);
		float_t top = 10.0*p;
		float_t bottom = (p+1.0)*(p+1.0);
		return top/bottom;
	};

	float_t* forward_pass(float_t* in)
	{
		inject_input(in, num_input);

		for (int out = 0; out < num_output; out++)
		{
			output[out] = 0.0f;

			for (int in = 0; in < num_input; in++)
			{


				output[out] += inputs[in] * weights[out][in];

			}
			output[out] = activation_function(output[out]);
		}

		return output;
	};

	float_t* back_pass(float_t* backprop, float_t learning_rate)
	{


		for (int out = 0; out < num_output; out++)
		{
			tmp[out] = activation_derivate(output[out])*backprop[out];
		}

		for (int out = 0; out < num_output; out++)
		{

			for (int in = 0; in < num_input; in++)
			{
				weights_delta[out][in] += learning_rate*(tmp[out] * inputs[in]);
			}

		}

		for (int in = 0; in < num_input; in++)
		{
			gradient[in] = 0.0f;
			for (int out = 0; out < num_output; out++)
			{
				gradient[in] += tmp[out] * weights[out][in];
			}
		}
		return gradient;

	};

	void set_momentum(float_t _decay_rate)
	{
		this->use_momentum = 1;
		this->decay_rate = _decay_rate;

		this->momentum = new float_t*[num_output];
		for (int i = 0; i < num_output; i++)
		{
			this->momentum[i] = new float_t[num_input];

			for (int j = 0; j < num_input; j++)
			{
				this->momentum[i][j] = 0.0;
			}
		}
	}

};
