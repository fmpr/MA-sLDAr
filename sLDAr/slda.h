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

#ifndef SLDA_H
#define SLDA_H
#include "settings.h"
#include "corpus.h"
#include <math.h>
//#include "cokus.h"

typedef struct {
    double * z_bar_m;
    double * z_bar_var;
} z_stat;

typedef struct {
    double ** word_ss;
    double * word_total_ss;
    int num_docs;
    z_stat * z_bar;

    // <---- ADICIONAR
    //double* b_stats;
    //double* v_stats;
    // ---->

    // for supervised LDA, by Jun Zhu
    double **covarmatrix; // E[zz^\top]
    double *ezy;          // y*E[z]
    double sqresponse;    // \sum_{d=1}^D y_d^2, the sum of square response variables


} suffstats;

class slda
{
public:
    slda();
    ~slda();
    void free_model();
    void init(double alpha_, int num_topics_, const corpus * c, double init_seed);
    void v_em(corpus * c, const settings * setting,
              const char * start, const char * directory);

    void save_model(const char * filename);
    void save_model_text(const char * filename);
    void load_model(const char * model_filename);
    void infer_only(corpus * c, const settings * setting,
                    const char * directory);

    suffstats * new_suffstats(int num_docs);
    void free_suffstats(suffstats * ss);
    void zero_initialize_ss(suffstats * ss);
    void random_initialize_ss(suffstats * ss, corpus * c);
    void corpus_initialize_ss(suffstats* ss, corpus * c);
    void load_model_initialize_ss(suffstats* ss, corpus * c);
    bool mle(suffstats * ss, bool bInit, const settings * setting);

    double doc_e_step(document* doc, double* gamma, double** phi, double** a, suffstats * ss, int eta_update, const settings * setting);
    double lda_inference(document* doc, double* var_gamma, double** phi, const settings * setting);
    double lda_compute_likelihood(document* doc, double** phi, double* var_gamma);
    double slda_inference(document* doc, double* var_gamma, double** phi, double** a, const settings * setting);
    double slda_compute_likelihood(document* doc, double** phi, double* var_gamma,  double** a);

    void save_gamma(char* filename, double** gamma, int num_docs);
    void write_word_assignment(FILE* f, document* doc, double** phi);
    void amatrix(document* doc, slda* model, double** phi, double** a);


public:
    double alpha; // the parameter for the dirichlet
    int num_topics;
    int num_classes;
    int size_vocab;
    double seed;
    double ** log_prob_w; //the log of the topic distribution
    double *eta;        // \eta
    double deltasq;     // \delta^2

    // for fast computing.
    double *oldphi_;
    double *digamma_gam_;
    double *phisumarry_;
    double *phiNotN_;
    double *dig_;
    double *arry_;

};

#endif // SLDA_H

