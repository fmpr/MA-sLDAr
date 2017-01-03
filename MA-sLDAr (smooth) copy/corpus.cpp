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

#include "corpus.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdlib>


corpus::corpus()
{
    num_docs = 0;
    size_vocab = 0;
    num_total_words = 0;
    num_annot = 0;
}

corpus::~corpus()
{
    for (int i = 0; i < num_docs; i ++)
    {
        document * doc = docs[i];
        delete doc;
    }
    docs.clear();

    num_docs = 0;
    size_vocab = 0;
    num_total_words = 0;
    num_annot = 0;
}

int corpus::read_data(char * data_filename,
                       char * answers_filename, int TRAIN, char * labels_filename) 

{
    int OFFSET = 0;
    int length = 0,  word = 0, i,
        n = 0, nd = 0, nw = 0, na = 0,first = 1, bytes_consumed = 0;
    double target = -1, value;

    double count = 0;
    char * buffer = new char[90000];
    char *pbuff;


    FILE * fileptr;
    fileptr = fopen(data_filename, "r");
    printf("\nreading data from %s\n", data_filename);
    nd = 0;
    nw = 0;

    while ((fscanf(fileptr, "%10d", &length) != EOF))
    {
        document * doc = new document(length);

        for (n = 0; n < length; n++)
        {
            fscanf(fileptr, "%10d:%lf", &word, &count);
            word = word - OFFSET;
            doc->words[n] = word;
            doc->counts[n] = count; 


            doc->total += count;

            if (word >= nw)
            {
                nw = word + 1;
            }
        }
        num_total_words += doc->total;
        docs.push_back(doc);
        nd++;
    }
    fclose(fileptr);
    num_docs = nd;
    size_vocab = nw;
    printf("number of docs  : %d\n", nd);
    printf("number of terms : %d\n", nw);
    printf("number of total words : %d\n", num_total_words);


    if(TRAIN)
    {
        fileptr = fopen(answers_filename, "r");
        printf("\nreading annotators answers from %s\n", answers_filename);




        for(nd = 0; nd < int(docs.size()); nd++) 
        {
            if (!fgets(buffer, sizeof(char)*90000, fileptr)) 
                break;
        
            pbuff = buffer;

            na = 0;

            while (first || na<num_annot) 
            {   

                if (*pbuff == '\n')
                    break;

                value = strtod(pbuff, &pbuff);

                if (value >= num_classes)
                    num_classes = value + 1;

                docs[nd]->answers.push_back(value);

                na++;
               
            }

            if(first)
            {
                num_annot = na;
                first = 0;

                delete [] buffer; 

                buffer = new char[num_annot*int(docs.size())];

            }
            else
                assert(na == num_annot);
            
        }
        

        delete [] buffer; 

        assert(nd == int(docs.size()));

        printf("number of annotators : %d\n\n", num_annot);

    }

    if(!TRAIN)
        labels_filename = answers_filename;

    int labels_file = 0;

    printf("LABELS FILE: %s \n%p\n", labels_filename, fopen(labels_filename, "r"));
    if((labels_filename!= NULL) && (fopen(labels_filename, "r") != NULL))
    {
        fileptr = fopen(labels_filename, "r");
        printf("\nreading targets from %s\n", labels_filename);
        nd = 0;
        while ((fscanf(fileptr, "%lf", &target) != EOF))
        {
            document * doc = docs[nd];
            doc->responseVar= target;
            printf("target: %f\n", target);

            nd ++;
        }
        assert(nd == int(docs.size()));

        labels_file = 1;
    }

    return labels_file;
  
}

int corpus::max_corpus_length() 
{
    int max_length = 0;

    for (int d = 0; d < num_docs; d++) 
    {
        if (docs[d]->length > max_length)
            max_length = docs[d]->length;
    }
    return max_length;
}
