#include <iostream>
#include <fstream>
#include <string>
#include <limits.h>
#include <ctime>
#include <stdlib.h>
#include <stdio.h>
#include <list>
#include <cmath>

#include "Networks.h"

typedef struct _frame
{
	_frame* next_frame;
	_frame* prev_frame;
	float_t* input;
	float_t* res;
} frame;


class grab_point
{
private:
	Network* nn;
	float_t player_pos_x, player_pos_y;
	float_t target_x, target_y;
	int count;
	frame* frames;
public:
	grab_point()
	{
		nn = new Network();
		nn->add_layer(new layer_fully_connected( 4, 10));
		nn->add_layer(new layer_fully_connected(10, 8));
		nn->add_layer(new layer_fully_connected(8,  2));
		nn->set_learning_rate(0.000001);


		target_x = 15.0;
		target_y = 7.0;
		player_pos_x = 0.0;
		player_pos_y = 0.0;
		count = 0;
		frames = nullptr;
	}

	~grab_point()
	{

	}

	int check_near(float_t margin)
	{
		if (abs(player_pos_x - target_x) > margin)
			return 0;

		if (abs(player_pos_y - target_y) > margin)
			return 0;

		return 1;
	}

	void restart()
	{
		player_pos_x = 0.0;
		player_pos_y = 0.0;
		count = 0;
	}

	void update(int max_count)
	{
		if (check_near(0.1) == 1)
			return;

		if (frames != nullptr)
		{
			frame* ptr = frames;
			while (ptr->next_frame != nullptr)
			{
				ptr = ptr->next_frame;
			}

			while (ptr->prev_frame->prev_frame != nullptr)
			{
				ptr = ptr->prev_frame;
				delete[] ptr->input;
				delete[] ptr->res;
				delete ptr->next_frame;
			}
			delete ptr;

		}
		frames = new frame;
		frames->next_frame = nullptr;
		frames->prev_frame = nullptr;
		frame* f_ptr = frames;

		for (int i = 0; ((max_count>i) && (check_near(0.1) == 0)); i++)
		{
			count++;
			float_t* tbp = new float_t[4];
			tbp[0] = player_pos_x;
			tbp[1] = player_pos_y;
			tbp[2] = target_x;
			tbp[3] = target_y;
			float_t* resultado = nn->predict(tbp);
			
			f_ptr->next_frame = new frame;
			f_ptr->next_frame->prev_frame = f_ptr;
			f_ptr = f_ptr->next_frame;
			f_ptr->input = tbp;
			f_ptr->res = resultado;
			f_ptr->next_frame = nullptr;
			player_pos_x += resultado[0];
			player_pos_y += resultado[1];

			//printf("x %.6f   y %.6f\n", player_pos_x, player_pos_y);
		}

		if (check_near(0.1) == 0)
		{
			float_t init_learn_rate = 0.15f;
			while (f_ptr->prev_frame->prev_frame != nullptr)
			{
				nn->set_learning_rate(init_learn_rate);
				f_ptr->res[0] = target_x - f_ptr->input[0];
				f_ptr->res[1] = target_y - f_ptr->input[1];
				nn->train(-1.0, 10, &f_ptr->input, &f_ptr->res, 1);
				
				//printf(".");
				f_ptr = f_ptr->prev_frame;
				init_learn_rate *= 0.99f;
			}
			//printf("\n");
		}

	}

	int get_count()
	{
		return count;
	}

};
