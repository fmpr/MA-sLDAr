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

#include "slda.h"
#include <time.h>
#include "utils.h"
#include "assert.h"
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <cmath> 
using namespace std;

const double PI = 3.14159265359;
const int NUM_INIT = 50;
const int LAG = 10;
const int LDA_INIT_MAX = 0;
const int MSTEP_MAX_ITER = 50;

slda::slda()
{
    //ctor
    alpha = 1.0;
    num_topics = 0;
    num_classes = 0;
    size_vocab = 0;

    log_prob_w = NULL;
    eta = NULL;
}

slda::~slda()
{
    free_model();
}

/*
 * init the model
 */

void slda::init(double alpha_, int num_topics_,
                const corpus * c, double init_seed)
{
    alpha = alpha_;
    num_topics = num_topics_;
    size_vocab = c->size_vocab;
    num_classes = c->num_classes;
    seed = init_seed;


    // MATRIX K*V INITIALIZATION
    log_prob_w = new double * [num_topics];
    for (int k = 0; k < num_topics; k++)
    {
        log_prob_w[k] = new double [size_vocab]; 
        memset(log_prob_w[k], 0, sizeof(double)*size_vocab);
    }


    //no need to train slda if we only have on class
    
    eta = new double [num_topics];
    memset(eta, 0, sizeof(double)*num_topics);


    oldphi_ = (double*)malloc(sizeof(double)*num_topics);
    digamma_gam_ = (double*)malloc(sizeof(double)*num_topics);
    phisumarry_ = (double*)malloc( sizeof(double) * num_topics );
    phiNotN_ = (double*)malloc( sizeof(double) * num_topics );
    dig_ = (double*)malloc( sizeof(double) * num_topics );
    arry_ = (double*)malloc( sizeof(double) * num_topics );

    deltasq = 0.001;

        
    
}

/*
 * free the model
 */

void slda::free_model()
{
    if (log_prob_w != NULL)
    {
        for (int k = 0; k < num_topics; k++)
        {
            delete [] log_prob_w[k];
        }
        delete [] log_prob_w;
        log_prob_w = NULL;
    }
    if (eta != NULL)
    {
        delete [] eta;
        eta = NULL;
    }
}

/*
 * save the model in the binary format
 */

void slda::save_model(const char * filename)
{
    FILE * file = NULL;
    file = fopen(filename, "wb");
    fwrite(&alpha, sizeof (double), 1, file);
    fwrite(&num_topics, sizeof (int), 1, file);
    fwrite(&size_vocab, sizeof (int), 1, file);
    fwrite(&num_classes, sizeof (int), 1, file);

    for (int k = 0; k < num_topics; k++)
    {
        fwrite(log_prob_w[k], sizeof(double), size_vocab, file);
    }

    fwrite(eta, sizeof(double), num_topics, file);

    

    fflush(file);
    fclose(file);
}

/*
 * load the model in the binary format
 */

void slda::load_model(const char * filename)
{
    FILE * file = NULL;
    file = fopen(filename, "rb");
    fread(&alpha, sizeof (double), 1, file);
    fread(&num_topics, sizeof (int), 1, file);
    fread(&size_vocab, sizeof (int), 1, file);
    fread(&num_classes, sizeof (int), 1, file);

    log_prob_w = new double * [num_topics];
    for (int k = 0; k < num_topics; k++)
    {
        log_prob_w[k] = new double [size_vocab];
        fread(log_prob_w[k], sizeof(double), size_vocab, file);
    }
    
    eta = new double [num_topics];
    fread(eta, sizeof(double), num_topics, file);

    

    fflush(file);
    fclose(file);
}

/*
 * save the model in the text format
 */

void slda::save_model_text(const char * filename)
{
    FILE * file = NULL;
    file = fopen(filename, "w");
    fprintf(file, "alpha: %lf\n", alpha);
    fprintf(file, "number of topics: %d\n", num_topics);
    fprintf(file, "size of vocab: %d\n", size_vocab);
    fprintf(file, "number of classes: %d\n", num_classes);

    fprintf(file, "betas: \n"); // in log space
    for (int k = 0; k < num_topics; k++)
    {
        for (int j = 0; j < size_vocab; j ++)
        {
            fprintf(file, "%lf ", log_prob_w[k][j]);
        }
        fprintf(file, "\n");
    }

    fprintf(file, "etas: \n");
    for (int j = 0; j < num_topics; j ++)
        fprintf(file, "%lf ", eta[j]);
    fprintf(file, "\n");
        
    

    fflush(file);
    fclose(file);
}

/*
 * create the data structure for sufficient statistic 
 */

suffstats * slda::new_suffstats(int num_docs)
{
    suffstats * ss = new suffstats;

    ss->num_docs = num_docs;
    
    ss->word_total_ss = new double [num_topics];
    
    memset(ss->word_total_ss, 0, sizeof(double)*num_topics);
    
    // MATRIX K*V INITIALIZATION
    ss->word_ss = new double * [num_topics];
    for (int k = 0; k < num_topics; k ++)
    {
        ss->word_ss[k] = new double [size_vocab];
        memset(ss->word_ss[k], 0, sizeof(double)*size_vocab);
    }

    // SUM 1->K
    int num_var_entries = num_topics*(num_topics+1)/2; 

    ss->z_bar =  new z_stat [num_docs];
    for (int d = 0; d < num_docs; d ++)
    {
        ss->z_bar[d].z_bar_m = new double [num_topics];
        ss->z_bar[d].z_bar_var = new double [num_var_entries];
        memset(ss->z_bar[d].z_bar_m, 0, sizeof(double)*num_topics);
        memset(ss->z_bar[d].z_bar_var, 0, sizeof(double)*num_var_entries);
    }

    // for sLDA only
    ss->covarmatrix = (double**)malloc(sizeof(double*)*num_topics);
    ss->ezy = (double*)malloc(sizeof(double) * num_topics);
    for ( int k=0; k<num_topics; k++ ) 
    {
        ss->ezy[k] = 0;
        ss->covarmatrix[k] = (double*)malloc(sizeof(double)*num_topics);
        for (int i=0; i<num_topics; i++ )
            ss->covarmatrix[k][i] = 0;
    }
    ss->sqresponse = 0;

    return(ss);
}


/*
 * initialize the sufficient statistics with zeros
 */

void slda::zero_initialize_ss(suffstats * ss)
{
    memset(ss->word_total_ss, 0, sizeof(double)*num_topics);
    for (int k = 0; k < num_topics; k ++)
    {
        memset(ss->word_ss[k], 0, sizeof(double)*size_vocab);
    }

    int num_var_entries = num_topics*(num_topics+1)/2;
    for (int d = 0; d < ss->num_docs; d ++)
    {
        memset(ss->z_bar[d].z_bar_m, 0, sizeof(double)*num_topics);
        memset(ss->z_bar[d].z_bar_var, 0, sizeof(double)*num_var_entries);
    }
    ss->num_docs = 0;


    // for sLDA only
    for ( int k=0; k<num_topics; k++ ) 
    {
        ss->ezy[k] = 0;
        for ( int i=0; i<num_topics; i++ )
            ss->covarmatrix[k][i] = 0;
    }

}


/*
 * initialize the sufficient statistics with random numbers 
 */

void slda::random_initialize_ss(suffstats * ss, corpus* c)
{
    int num_docs = ss->num_docs;

    // CREATE RANDOM GENERATOR
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);

    //time_t seed;
    //time(&seed);

    // RANDOM GENERATOR INITIALIZATION
    gsl_rng_set(rng, (long) seed);

    int k, w, d, j, idx;
    for (k = 0; k < num_topics; k++)
    {
        for (w = 0; w < size_vocab; w++)
        {
            ss->word_ss[k][w] = 1.0/size_vocab + 0.1*gsl_rng_uniform(rng);
            ss->word_total_ss[k] += ss->word_ss[k][w];
        }
    }

    for (d = 0; d < num_docs; d ++)
    {
        document * doc = c->docs[d];

        double total = 0.0;
        for (k = 0; k < num_topics; k ++)
        {
            ss->z_bar[d].z_bar_m[k] = gsl_rng_uniform(rng);
            total += ss->z_bar[d].z_bar_m[k];
        }

        for (k = 0; k < num_topics; k ++)
        {
            ss->z_bar[d].z_bar_m[k] /= total; // SUM OF ALL s->z_bar[d].z_bar_m[k] = 1
        }


        // ...nao percebo
        for (k = 0; k < num_topics; k ++)
        {
            for (j = k; j < num_topics; j ++)
            {
                idx = map_idx(k, j, num_topics);

                if (j == k)
                    ss->z_bar[d].z_bar_var[idx] = ss->z_bar[d].z_bar_m[k] / (double)(doc->total);
                else
                    ss->z_bar[d].z_bar_var[idx] = 0.0;

                ss->z_bar[d].z_bar_var[idx] -=
                    ss->z_bar[d].z_bar_m[k] * ss->z_bar[d].z_bar_m[j] / (double)(doc->total); 
            }
        }
    }

    gsl_rng_free(rng);
}

void slda::corpus_initialize_ss(suffstats* ss, corpus* c)
{
    int num_docs = ss->num_docs;
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);
    //time_t seed;
    //time(&seed);
    gsl_rng_set(rng, (long) seed);
    int k, n, d, j, idx, i, w;

    for (k = 0; k < num_topics; k++)
    {
        for (i = 0; i < NUM_INIT; i++)
        {
            d = (int)(floor(gsl_rng_uniform(rng) * num_docs));
            printf("initialized with document %d\n", d);
            document * doc = c->docs[d];
            for (n = 0; n < doc->length; n++)
            {
                ss->word_ss[k][doc->words[n]] += doc->counts[n];
            }
        }
        for (w = 0; w < size_vocab; w++)
        {
            ss->word_ss[k][w] = 2*ss->word_ss[k][w] + 5 + gsl_rng_uniform(rng);
            ss->word_total_ss[k] = ss->word_total_ss[k] + ss->word_ss[k][w];
        }
    }

    for (d = 0; d < num_docs; d ++)
    {
        document * doc = c->docs[d];

        double total = 0.0;
        for (k = 0; k < num_topics; k ++)
        {
            ss->z_bar[d].z_bar_m[k] = gsl_rng_uniform(rng);
            total += ss->z_bar[d].z_bar_m[k];
        }
        for (k = 0; k < num_topics; k ++)
        {
            ss->z_bar[d].z_bar_m[k] /= total;
        }
        for (k = 0; k < num_topics; k ++)
        {
            for (j = k; j < num_topics; j ++)
            {
                idx = map_idx(k, j, num_topics);
                if (j == k)
                    ss->z_bar[d].z_bar_var[idx] = ss->z_bar[d].z_bar_m[k] / (double)(doc->total);
                else
                    ss->z_bar[d].z_bar_var[idx] = 0.0;

                ss->z_bar[d].z_bar_var[idx] -=
                    ss->z_bar[d].z_bar_m[k] * ss->z_bar[d].z_bar_m[j] / (double)(doc->total);
            }
        }
    }
    gsl_rng_free(rng);
}

void slda::load_model_initialize_ss(suffstats* ss, corpus * c)
{
    int num_docs = ss->num_docs;                                                                         
    for (int d = 0; d < num_docs; d ++)       
    {                                                                                                    
       document * doc = c->docs[d];

    }     
}

void slda::free_suffstats(suffstats * ss)
{
    delete [] ss->word_total_ss;

    for (int k = 0; k < num_topics; k ++)
    {
        delete [] ss->word_ss[k];
    }
    delete [] ss->word_ss;

    for (int d = 0; d < ss->num_docs; d ++)
    {
        delete [] ss->z_bar[d].z_bar_m;
        delete [] ss->z_bar[d].z_bar_var;
    }
    delete [] ss->z_bar;

    delete ss;
}

void slda::v_em(corpus * c, const settings * setting,
                const char * start, const char * directory)
{
    char filename[100];
    int max_length = c->max_corpus_length();
    double **var_gamma, **phi, **lambda;
    double likelihood, likelihood_old = 0, converged = 1;
    int d, n, i;
    double L2penalty = setting->PENALTY;
   
    // allocate variational parameters
    var_gamma = new double * [c->num_docs];
    for (d = 0; d < c->num_docs; d++)
        var_gamma[d] = new double [num_topics];

    phi = new double * [max_length];
    for (n = 0; n < max_length; n++)
        phi[n] = new double [num_topics];

    double **a = (double**)malloc(sizeof(double*) * num_topics);
    for ( int k=0; k<num_topics; k++ )
        a[k] = (double*)malloc(sizeof(double) * num_topics);


    printf("initializing ...\n");
    suffstats * ss = new_suffstats(c->num_docs);

    if (strcmp(start, "seeded") == 0)
    {
        corpus_initialize_ss(ss, c);
        mle(ss, true, setting);
    }
    else if (strcmp(start, "random") == 0)
    {
        random_initialize_ss(ss, c);
        mle(ss, true, setting);
    }
    else
    {
        load_model(start);
        load_model_initialize_ss(ss, c);
    }


    double dmean = 0;
    for ( d=0; d<c->num_docs; d++ )
        dmean += c->docs[d]->responseVar / c->num_docs;
   
   /* deltasq = 0;
    for ( d=0; d<c->num_docs; d++ )
        deltasq += (c->docs[d]->responseVar - dmean) * (c->docs[d]->responseVar - dmean)
        / c->num_docs;*/

    FILE * likelihood_file = NULL;
    sprintf(filename, "%s/likelihood.dat", directory);
    likelihood_file = fopen(filename, "w");

    int ETA_UPDATE = 0;

    i = 0;
    while (((converged < 0) || (converged > setting->EM_CONVERGED) || (i <= LDA_INIT_MAX+2)) && (i <= setting->EM_MAX_ITER))
    {
        printf("**** em iteration %d ****\n", ++i);
      
        likelihood = 0;
      
        zero_initialize_ss(ss);
      
        if (i > LDA_INIT_MAX) 
            ETA_UPDATE = 1;
      

        // e-step
        printf("**** e-step ****\n");
        for (d = 0; d < c->num_docs; d++)
        {   
            if ((d % 100) == 0) 
                printf("document %d\n", d);

            likelihood += doc_e_step(c->docs[d], var_gamma[d], phi, a, ss, ETA_UPDATE, setting);
        }

            // m-step
        printf("**** m-step ****\n");
        mle(ss, false, setting);

        // check for convergence
        converged = fabs((likelihood_old - likelihood) / (likelihood_old));
        likelihood_old = likelihood;

        printf("Likelihood: %8.5f\n", likelihood);

        // output model and likelihood
        fprintf(likelihood_file, "%10.10f\t%5.5e\n", likelihood, converged);
        fflush(likelihood_file);
        if ((i % LAG) == 0)
        {
            sprintf(filename, "%s/%03d.model", directory, i);
            save_model(filename);
            sprintf(filename, "%s/%03d.model.text", directory, i);
            save_model_text(filename);
            sprintf(filename, "%s/%03d.gamma", directory, i);
            save_gamma(filename, var_gamma, c->num_docs);
        }
    }

    // output the final model
    sprintf(filename, "%s/final.model", directory);
    save_model(filename);
    sprintf(filename, "%s/final.model.text", directory);
    save_model_text(filename);
    sprintf(filename, "%s/final.gamma", directory);
    save_gamma(filename, var_gamma, c->num_docs);


    fclose(likelihood_file);
    FILE * w_asgn_file = NULL;
    sprintf(filename, "%s/word-assignments.dat", directory);
    w_asgn_file = fopen(filename, "w");

    for (d = 0; d < c->num_docs; d ++)
    {
        //final inference
        if ((d % 100) == 0) printf("final e step document %d\n", d);
        likelihood += slda_inference(c->docs[d], var_gamma[d], phi, a, setting);
        write_word_assignment(w_asgn_file, c->docs[d], phi);

    }
    fclose(w_asgn_file);

    free_suffstats(ss);
    for (d = 0; d < c->num_docs; d++)
        delete [] var_gamma[d];
    delete [] var_gamma;

    for (n = 0; n < max_length; n++)
        delete [] phi[n];
    delete [] phi;
}


bool slda::mle(suffstats * ss, bool bInit, const settings * setting) // M-STEP
{
    int k, w;

    // F: update beta; this is simply given by the number of times the word n appears associated with topic k, 
    // divided the number of times topic k appear associated with any word (i.e. normalized);
    // in log-space, this becomes: log(ss->word_ss[k][w]) - log(ss->word_total_ss[k]);
    for (k = 0; k < num_topics; k++)
    {
        for (w = 0; w < size_vocab; w++)
        {
            if (ss->word_ss[k][w] > 0)
                log_prob_w[k][w] = log(ss->word_ss[k][w]) - log(ss->word_total_ss[k]);
            else
                log_prob_w[k][w] = -100.0;
        }
    }
    bool bRes = true;
    if ( !bInit ) 
    {
        // \eta for supervised LDA (Blei & McAuliffe, 2007)
        double **inversmatrix = (double**)malloc(sizeof(double*)*num_topics);
        for ( int i=0; i<num_topics; i++ )
            inversmatrix[i] = (double*)malloc(sizeof(double)*num_topics);

        bRes = inverse( ss->covarmatrix, inversmatrix, num_topics );
        matrixprod(inversmatrix, ss->ezy, eta, num_topics);

    }  


    for(int k = 0; k<num_topics; k++)
        printf("eta: %f\n", eta[k]);


    return bRes;

}


// ALTERAR
double slda::doc_e_step(document* doc, double* gamma, double** phi, double** a,
                        suffstats * ss, int eta_update, const settings * setting) // E-STEP
{
    double likelihood = 0.0;

    if (eta_update == 1)
        likelihood = slda_inference(doc, gamma, phi, a, setting);
    else
        likelihood = lda_inference(doc, gamma, phi, setting);

    int d = ss->num_docs;

    int n, k, i, idx, m;

    // update sufficient statistics

    double *covPtr=NULL;
    for (k = 0; k < num_topics; k++) 
    {

    
        // suff-stats for supervised LDA
        covPtr = ss->covarmatrix[k];
        for ( n=0; n<num_topics; n++ )
            covPtr[n] += a[k][n];
    }

    for (k = 0; k < num_topics; k++)
    {
        double dVal = 0;

        for (n = 0; n < doc->length; n++)
        {
            // F: compute the number of times word n appears associated with topic k (KxN matrix);
            // these are soft counts! since phi is not zeros and ones (it's a probability of assigning topic k to word n);
            // notice how this corresponds to eq. 9 of Blei2003 (except for the outer sum over all documents)
            ss->word_ss[k][doc->words[n]] += doc->counts[n]*phi[n][k];
            // F: compute the number of times topic k appears associated with any word (K vector)
            ss->word_total_ss[k] += doc->counts[n]*phi[n][k];

            //statistics for each document of the supervised part
            // F: the mean and variance of bar{z} for each document
            ss->z_bar[d].z_bar_m[k] += doc->counts[n] * phi[n][k]; //mean
           
            for (i = k; i < num_topics; i ++) //variance
            {
                idx = map_idx(k, i, num_topics);
               
                if (i == k)
                    ss->z_bar[d].z_bar_var[idx] +=
                        doc->counts[n] * doc->counts[n] * phi[n][k]; 

                ss->z_bar[d].z_bar_var[idx] -=
                    doc->counts[n] * doc->counts[n] * phi[n][k] * phi[n][i];
            }

            dVal += phi[n][k] * doc->counts[n] / doc->total;

        }

        ss->ezy[k] += dVal * doc->responseVar;

    }

    for (k = 0; k < num_topics; k++)
    {
        ss->z_bar[d].z_bar_m[k] /= (double)(doc->total);
    }

    for (i = 0; i < num_topics*(num_topics+1)/2; i ++)
    {
        ss->z_bar[d].z_bar_var[i] /= (double)(doc->total * doc->total);
    }

    ss->num_docs = ss->num_docs + 1; //because we need it for store statistics for each docs

    return (likelihood);
}

double slda::lda_inference(document* doc, double* var_gamma, double** phi, const settings * setting)
{
    int k, n, var_iter;
    double converged = 1, phisum = 0, likelihood = 0, likelihood_old = 0;

    double *oldphi = new double [num_topics];
    double *digamma_gam = new double [num_topics];

    // compute posterior dirichlet
    // F: initilization of the standard LDA algorithm
    for (k = 0; k < num_topics; k++)
    {
        // F: initilize gamma_k = alpha + N/K (Step 2 of Figure 6 of Blei2003)
        var_gamma[k] = alpha + (doc->total/((double) num_topics)); 
        digamma_gam[k] = digamma(var_gamma[k]);

        for (n = 0; n < doc->length; n++)
            // F: initilize phi_nk = 1/K (Step 1 of Figure 6 of Blei2003)
            phi[n][k] = 1.0/num_topics; 
    }


    var_iter = 0;

    while (converged > setting->VAR_CONVERGED && (var_iter < setting->VAR_MAX_ITER || setting->VAR_MAX_ITER == -1))
    {
        var_iter++;
        for (n = 0; n < doc->length; n++) // STEP 4 FIGURE 6 OF LDA
        {
            phisum = 0;

            for (k = 0; k < num_topics; k++) // STEP 5 FIGURE 6 OF LDA
            {
                oldphi[k] = phi[n][k];

                // F: standard phi update from Figure 6, Step 6 of Blei2003, but IN LOG SPACE! hence the product becomes a sum...
                phi[n][k] = digamma_gam[k] + log_prob_w[k][doc->words[n]]; 

                // F: this computes the normalization constant of phi; for example the digamma(sum_j phi_j) comes from here...
                if (k > 0)
                    phisum = log_sum(phisum, phi[n][k]); 
                else
                    phisum = phi[n][k]; // note, phi is in log space
            }

            // F: normalize the phis and update the gammas (Step 8 from Figure 6 of Blei2003)
            for (k = 0; k < num_topics; k++)
            {
                // F: normalize and move back from log space, by exponentiating...
                phi[n][k] = exp(phi[n][k] - phisum); // normalize

                // F: this update is in a sequencial form; to verify that it is correct notice the following:
                // gamma^(t+1) = alpha + sum_n phi_n^(t+1) = alpha + sum_n phi_n^(t+1) + sum_n phi_n^t - sum_n phi_n^t = gamma^t + sum(phi_n^(t+1) - phi_n^t)
                var_gamma[k] = var_gamma[k] + doc->counts[n]*(phi[n][k] - oldphi[k]);
                
                // !!! a lot of extra digamma's here because of how we're computing it
                // !!! but its more automatically updated too.
                digamma_gam[k] = digamma(var_gamma[k]);
            }
        }

        likelihood = lda_compute_likelihood(doc, phi, var_gamma);
        assert(!isnan(likelihood));
        converged = (likelihood_old - likelihood) / likelihood_old;
        likelihood_old = likelihood;


    }

    delete [] oldphi;
    delete [] digamma_gam;

    return likelihood;
}

double slda::lda_compute_likelihood(document* doc, double** phi, double* var_gamma)
{
    double likelihood = 0, digsum = 0, var_gamma_sum = 0;
    double *dig = new double [num_topics];
    int k, n;
    double alpha_sum = num_topics * alpha;
    
    for (k = 0; k < num_topics; k++)
    {
        dig[k] = digamma(var_gamma[k]);
        var_gamma_sum += var_gamma[k];
    }
    
    digsum = digamma(var_gamma_sum);

    likelihood = lgamma(alpha_sum) - lgamma(var_gamma_sum);


    for (k = 0; k < num_topics; k++)
    {
        likelihood += - lgamma(alpha) + (alpha - 1)*(dig[k] - digsum) +
                      lgamma(var_gamma[k]) - (var_gamma[k] - 1)*(dig[k] - digsum);


        for (n = 0; n < doc->length; n++)
        {
            if (phi[n][k] > 0)
            {
                likelihood += doc->counts[n]*(phi[n][k]*((dig[k] - digsum) -
                                              log(phi[n][k]) + log_prob_w[k][doc->words[n]]));

            }
        }
    }

    delete [] dig;
    return likelihood;
}

double slda::slda_compute_likelihood(document* doc, double** phi, double* var_gamma, double ** a)
{
    double likelihood = 0, digsum = 0, var_gamma_sum = 0, t = 0.0, t1 = 0.0, t2 = 0.0;
    double * dig = new double [num_topics];
    int k, n, l;
    int flag;
    double alpha_sum = num_topics * alpha;
  
    for (k = 0; k < num_topics; k++)
    {
        dig[k] = digamma(var_gamma[k]);
        var_gamma_sum += var_gamma[k];
    }
  
    digsum = digamma(var_gamma_sum);

    likelihood = lgamma(alpha_sum) - num_topics*lgamma(alpha) - lgamma(var_gamma_sum);
    for (k = 0; k < num_topics; k++)
    {
        likelihood += (alpha - 1)*(dig[k] - digsum) + lgamma(var_gamma[k]) - (var_gamma[k] - 1)*(dig[k] - digsum);

        double dVal = 0;
        for (n = 0; n < doc->length; n++)
        {
            if (phi[n][k] > 0)
                likelihood += doc->counts[n]*(phi[n][k]*((dig[k] - digsum) - log(phi[n][k]) + log_prob_w[k][doc->words[n]]));

            dVal += phi[n][k] * doc->counts[n] /  (double)(doc->total);

        }    

        likelihood += (doc->responseVar * eta[k] * dVal) / deltasq;
    }

    likelihood -= 0.5 * log(deltasq * 2 * PI);
    likelihood -= (doc->responseVar * doc->responseVar) / ( 2 * deltasq );

    double *arry = (double*)malloc(sizeof(double)*num_topics);
    matrixprod(eta, a, arry, num_topics);
   
    double dVal = dotprod(arry, eta, num_topics);

    likelihood -= dVal / ( 2 * deltasq );


    free(arry);    
    free(dig);

    return(likelihood);
}

double slda::slda_inference(document* doc, double* var_gamma, double** phi, double ** a, const settings * setting)
{
    int k, n, var_iter, l;
    int FP_MAX_ITER = 10;
    int fp_iter = 0;
    double converged = 1, phisum = 0, likelihood = 0, likelihood_old = 0;
    double * oldphi = new double [num_topics];
    double * digamma_gam = new double [num_topics];
    double * sf_params = new double [num_topics];

    // compute posterior dirichlet

    // mudei mariana
    double *phisumarry = (double*)malloc( sizeof(double) * num_topics ); //phisumarry_;
    double *phiNotN = (double*)malloc( sizeof(double) * num_topics ); //phiNotN_;
  
    for (k = 0; k < num_topics; k++)
    {
        var_gamma[k] = alpha + (doc->total/((double) num_topics));
        digamma_gam[k] = digamma(var_gamma[k]);
      
        double tmp = 0;
        for (n = 0; n < doc->length; n++)
        {
            phi[n][k] = 1.0/(double)(num_topics);
            tmp += phi[n][k] * doc->counts[n];
        }
        phisumarry[k] = tmp;
    }

    var_iter = 0;

    while ((converged > setting->VAR_CONVERGED) && ((var_iter < setting->VAR_MAX_ITER) || (setting->VAR_MAX_ITER == -1)))    
    {
        var_iter ++;
        for (n = 0; n < doc->length; n++)
        {

            /* \eta^\top \phi_{-n} */
            for ( k = 0; k < num_topics; k++ ) 
                phiNotN[k] = phisumarry[k] - phi[n][k]*doc->counts[n];

            double dProd = dotprod(eta, phiNotN, num_topics);
           // printf("dProd: %f\n", dProd);
            double Nsigma2 = deltasq * doc->total; /* N \delta^2 */

            phisum = 0; 
            for (k = 0; k < num_topics; k++)
            {
                oldphi[k] = phi[n][k];
                
                /* update the phi: add additional terms here for supervised LDA */
                // <---- MUDAR                     (N^d*....
                
                double last_term_phi =  doc->counts[n]*(2*eta[k]*dProd + eta[k]*eta[k]*doc->counts[n])  / (2*Nsigma2*doc->total);
              //  printf("last_term_phi (%f)=   doc->counts[n]*(2*eta[k](%f) * dProd(%f) + eta[k]*eta[k]*doc->counts[n])  / (2*Nsigma2(%f)*doc->total)\n", last_term_phi, eta[k], dProd, Nsigma2);

                
           //     double last_term_phi = doc->counts[n]*doc->counts[n]*(2*eta[k]*dProd + eta[k]*eta[k])   / (2*Nsigma2*doc->total);

                // ---->
                // <---- MUDAR               exp_log_beta[k][doc->words[n]]
                phi[n][k] = log_prob_w[k][doc->words[n]] + digamma_gam[k]  // the following two terms for sLDA
                    //              *doc->m....
                    + (eta[k]*doc->responseVar*doc->counts[n]) / Nsigma2
                    - last_term_phi;
                // ---->

               // printf("phi(%f)=  digamma_gam (%f) + log_prob_w (%f) + (eta[k]*doc->responseVar(%f)*doc->counts[n])(%f) / Nsigma2(%f) - last_term_phi (%f)\n", phi[n][k], digamma_gam[k], log_prob_w[k][doc->words[n]], eta[k], doc->responseVar*doc->counts[n], Nsigma2, last_term_phi);


                if (k > 0) phisum = log_sum(phisum, phi[n][k]);
                else       phisum = phi[n][k]; // note, phi is in log space


            }

            // update gamma and normalize phi
            for (k = 0; k < num_topics; k++)
            {
                phi[n][k] = exp(phi[n][k] - phisum);
                var_gamma[k] = var_gamma[k] + doc->counts[n]*(phi[n][k] - oldphi[k]);
                // !!! a lot of extra digamma's here because of how we're computing it
                // !!! but its more automatically updated too.
                digamma_gam[k] = digamma(var_gamma[k]);

                phisumarry[k] = phiNotN[k] + phi[n][k] * doc->counts[n];

            }


        }

     
        /* compute the E[zz^\top] matrix. */
        amatrix(doc, this, phi, a);

        likelihood = slda_compute_likelihood(doc, phi, var_gamma, a);
        assert(!isnan(likelihood));
        converged = (likelihood_old - likelihood) / likelihood_old;
        likelihood_old = likelihood;

    }


    free(oldphi);
    free(digamma_gam);
    free(phisumarry);
    free(phiNotN);

    return(likelihood);

}

void slda::amatrix(document* doc, slda* model, double** phi, double** a)
{
    for ( int k=0; k<model->num_topics; k++ ) {
        for ( int i=0; i<model->num_topics; i++ ) 
            a[k][i] = 0;
    }

    double dnorm = doc->total * doc->total;
    for ( int n=0; n<doc->length; n++ )
    {
        // diag{phi}
        for ( int k=0; k<model->num_topics; k++ ) {
            a[k][k] += (phi[n][k] * doc->counts[n] * doc->counts[n] ) / dnorm;
        }

        for ( int m=n+1; m<doc->length /*&& m != n*/; m++ )
        {
            double dfactor = doc->counts[n] * doc->counts[m] / dnorm;
            addmatrix2(a, phi[n], phi[m], model->num_topics, dfactor);
        }
    }
}

void slda::infer_only(corpus * c, const settings * setting, const char * directory)
{
    FILE* fileptr;
    char filename[100];
    int i, d, n;
    double **var_gamma, likelihood, **phi;
    document* doc;

    var_gamma = (double**)malloc(sizeof(double*)*(c->num_docs));
    for (i = 0; i < c->num_docs; i++)
        var_gamma[i] = (double*)malloc(sizeof(double)*num_topics);
    
    double **a = (double**)malloc(sizeof(double*)*num_topics);
    for ( int k=0; k<num_topics; k++ )
        a[k] = (double*)malloc(sizeof(double) * num_topics);
    int nMaxLength = c->max_corpus_length();
    phi = (double**) malloc(sizeof(double*) * nMaxLength );
    for (n = 0; n < nMaxLength; n++) {
        phi[n] = (double*) malloc(sizeof(double) * num_topics);
    }

    int nAcc = 0;
    double dUniformPhiVal = 1.0 / (double) num_topics, *phiPtr=NULL;
    sprintf(filename, "%s/evl-lda-lhood.dat", directory);
    fileptr = fopen(filename, "w");


    for(int k = 0; k<num_topics; k++)
        printf("eta: %f\n", eta[k]);


    for (d = 0; d < c->num_docs; d++)
    {
        if (((d % 100) == 0) && (d>0)) printf("document %d\n",d);

        doc = c->docs[d];
        // initialize to uniform distrubtion
        for (n = 0; n < doc->length; n++) 
        {
            phiPtr = phi[n];
            for ( int k=0; k<num_topics; k++ )
                phiPtr[k] = dUniformPhiVal;
        }

        likelihood = lda_inference(doc, var_gamma[d], phi, setting);

        // do prediction
        doc->testresponseVar = 0;
        for ( int k=0; k<num_topics; k++ ) 
        {
            double dVal = 0;
            for ( int n=0; n<doc->length; n++ )
            {
                dVal += phi[n][k] * doc->counts[n] / doc->total;
            //  printf("phi[%d][%d] = %f * doc->counts[%d] = %d\n", n, k, phi[n][k], n, doc->counts[n]);

            }
            doc->testresponseVar += dVal * eta[k];
        }
        doc->likelihood = likelihood;

        if ( doc->testresponseVar > 0.5 && doc->responseVar > 0.5 ) nAcc ++;
        if ( doc->testresponseVar <= 0.5 && doc->responseVar < 0.5 ) nAcc ++;

        fprintf(fileptr, "%5.5f\n", likelihood);
    }
    fclose(fileptr);
    sprintf(filename, "%s/evl-gamma.dat", directory);
    save_gamma(filename, var_gamma, num_topics);

    // save the prediction performance
    sprintf(filename, "%s/evl-performance.dat", directory);


    double dmean = 0;
    double sumlikelihood = 0;
    int nterms = 0;
    double sumavglikelihood = 0;

    for ( int d=0; d<c->num_docs; d++ ) 
    {
        dmean += c->docs[d]->responseVar / c->num_docs;
        sumlikelihood += c->docs[d]->likelihood;
        nterms += c->docs[d]->total;
        sumavglikelihood += c->docs[d]->likelihood / c->docs[d]->total;
    }


    double perwordlikelihood1 = sumlikelihood / nterms;
    double perwordlikelihood2 = sumavglikelihood / c->num_docs;

    double ssd = 0;
    for ( int d=0; d<c->num_docs; d++ ) 
        ssd += (c->docs[d]->responseVar - dmean ) * (c->docs[d]->responseVar - dmean);

    double sum_dif_sq = 0;
    double sum_dif = 0;
    double sum_dif_mean = 0;
    double sum_dif_mean_sq = 0;

    for ( int d=0; d<c->num_docs; d++ )
    {
        sum_dif_sq += ( c->docs[d]->responseVar - c->docs[d]->testresponseVar )
                * ( c->docs[d]->responseVar - c->docs[d]->testresponseVar );
        sum_dif += std::abs(c->docs[d]->responseVar - c->docs[d]->testresponseVar);
    }


    for ( int d=0; d<c->num_docs; d++ )
    {
        sum_dif_mean += std::abs(c->docs[d]->responseVar - dmean);
        sum_dif_mean_sq += (c->docs[d]->responseVar - dmean)* (c->docs[d]->responseVar - dmean);
    }

    double predictiver2 = 1.0 - sum_dif_sq / ssd;

    printf("Predictive R2: %5.10f\n", predictiver2);
    fileptr = fopen(filename, "w");
    fprintf(fileptr, "MAE: %5.10f\n", sum_dif / c->num_docs);
    fprintf(fileptr, "RMSE: %5.10f\n", sqrt(sum_dif_sq / c->num_docs));
    fprintf(fileptr, "RAE: %5.10f\n", sum_dif / (sum_dif_mean));
    fprintf(fileptr, "RRSE: %5.10f\n", sqrt(sum_dif_sq / (sum_dif_mean_sq)));

    printf("MAE: %5.10f\n", sum_dif / c->num_docs);
    printf("RMSE: %5.10f\n", sqrt(sum_dif_sq / c->num_docs));
    printf("RAE: %5.10f\n", sum_dif / (sum_dif_mean));
    printf("RRSE: %5.10f\n", sqrt(sum_dif_sq / (sum_dif_mean_sq)));

    fprintf(fileptr, "predictive R^2: %5.10f\n", predictiver2 );
    fprintf(fileptr, "perword likelihood1: %5.10f\n", perwordlikelihood1);
    fprintf(fileptr, "perword likelihood2: %5.10f\n", perwordlikelihood2);


    fclose(fileptr);


    sprintf(filename, "%s/inf-gamma.dat", directory);
    save_gamma(filename, var_gamma, c->num_docs);

    for (d = 0; d < c->num_docs; d++)
        delete [] var_gamma[d];
    delete [] var_gamma;

    for (n = 0; n < nMaxLength; n++)
        delete [] phi[n];
    delete [] phi;

}

void slda::save_gamma(char* filename, double** gamma, int num_docs)
{
    int d, k;

    FILE* fileptr = fopen(filename, "w");
    for (d = 0; d < num_docs; d++)
    {
        fprintf(fileptr, "%5.10f", gamma[d][0]);
        for (k = 1; k < num_topics; k++)
            fprintf(fileptr, " %5.10f", gamma[d][k]);
        fprintf(fileptr, "\n");
    }
    fclose(fileptr);
}

void slda::write_word_assignment(FILE* f, document* doc, double** phi)
{
    int n;

    fprintf(f, "%03d", doc->length);
    for (n = 0; n < doc->length; n++)
    {
        fprintf(f, " %04d:%02d", doc->words[n], argmax(phi[n], num_topics));
    }
    fprintf(f, "\n");
    fflush(f);

}
