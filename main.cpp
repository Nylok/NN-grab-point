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
#include "grab_point.h"

void main()
{
	int sd = time(NULL);
	//int sd = 5864;
	

	srand(sd);
	printf("Here seed %d\n",sd);
	

	grab_point* gp = new grab_point();

	long itime = clock();
	
	for (int i = 0; i < 5000; i++)
	{
		gp->restart();
		gp->update(50);
	}
	printf("count %d\n", gp->get_count());

	printf("\n\ndelta time %d\n", clock() - itime);
};
