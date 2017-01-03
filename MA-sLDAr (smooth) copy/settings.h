// (C) Copyright 2009, Chong Wang, David Blei and Li Fei-Fei

// written by Chong Wang, chongw@cs.princeton.edu

// This file is part of slda.

// slda is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// slda is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
#ifndef SETTINGS_H
#define SETTINGS_H
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

struct settings
{
    float VAR_CONVERGED;
    int   VAR_MAX_ITER;
    float EM_CONVERGED;
    int   EM_MAX_ITER;
    int   ESTIMATE_ALPHA;
    char  ANN_QUAL_FILE[100];
    char  LABELS_TRN_FILE[1000];
    float PENALTY;
    double LAMBDA_SMOOTHER;

    int read_settings(char* filename)
    {




        FILE * fileptr;
        char alpha_action[100];
        char ann_qual_file[1000];
        char labels_train[1000];



        fileptr = fopen(filename, "r");
        fscanf(fileptr, "var max iter %d\n", &this->VAR_MAX_ITER);
        fscanf(fileptr, "var convergence %f\n", &this->VAR_CONVERGED);
        fscanf(fileptr, "em max iter %d\n", &this->EM_MAX_ITER);
        fscanf(fileptr, "em convergence %f\n", &this->EM_CONVERGED);
        fscanf(fileptr, "L2 penalty %f\n", &this->PENALTY);

        fscanf(fileptr, "alpha %s\n", alpha_action);
        if (strcmp(alpha_action, "fixed") == 0)
        {
            this->ESTIMATE_ALPHA = 0;
            printf("alpha is fixed ...\n");
        }
        else
        {
            this->ESTIMATE_ALPHA = 1;
            printf("alpha is esimated ...\n");
        }




        fscanf(fileptr, "labels train file %s\n", labels_train);
        strcpy(this->LABELS_TRN_FILE, labels_train);
        printf("labels train file %s\n", this->LABELS_TRN_FILE);


        fscanf(fileptr, "annotators quality file %s\n", ann_qual_file);

        printf("file annotators %s\n", ann_qual_file);
        if((fopen(ann_qual_file, "r") == NULL) )
        {
            printf("Error: the file of the annotators quality isn't valid\n");
            return 0;
        }

    
        strcpy(this->ANN_QUAL_FILE, ann_qual_file);
        printf("annotators quality file %s\n", this->ANN_QUAL_FILE);

        fscanf(fileptr, "lambda smoother %lf", &this->LAMBDA_SMOOTHER);

        fclose(fileptr);
        
        printf("var max iter %d\n", this->VAR_MAX_ITER);
        printf("var convergence %.2E\n", this->VAR_CONVERGED);
        printf("em max iter %d\n", this->EM_MAX_ITER);
        printf("em convergence %.2E\n", this->EM_CONVERGED);
        printf("L2 penalty %.2E\n", this->PENALTY);
        printf("Lambda smoother %f\n", this->LAMBDA_SMOOTHER);

        return 1;
    }
};

#endif // SETTINGS_H

