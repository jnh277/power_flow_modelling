data {
  int<lower=0> T;           // number of timesteps
  int<lower=0> N;           // number of busses (not including slack)
  matrix[N,T] y_mag;           // measured bus voltage magnitude
  matrix[N,T] y_p;            // measured bus active power
  matrix[N,T] y_q;            // measured bus reactive power
  real real_zpk;
  real imag_zpk;
//  complex Ypk;          // conductance per km
}
transformed data {
    complex Zpk = to_complex(real_zpk, imag_zpk);
}

parameters {
    array[N] real Vreal;
    array[N] real Vimag;
    vector<lower=0>[N] l;            // line lengths (km)
    real<lower=1e-6> sig_e;     // measurement standard deviation
}
transformed parameters {
    array[N] complex V;
    array[N] complex I;
    array[N] complex shat;          // load power estimate
    array[N] complex Y;             // line conductance

    // get conductance ??
    for (n in 1:N){
        V[n] = to_complex(Vreal[n], Vimag[n]);
        Y[n] = 1/(Zpk*l[n]);
    }

    // work out branch currents
    I[1] = Y[1] * (1. - V[1]);
    I[2] = Y[2] * (V[1] - V[2]);
    I[3] = Y[3] * (V[1] - V[3]);
    I[4] = Y[4] * (V[3] - V[4]);

    // kcl to get powers
    shat[1] = V[1] * conj(I[1] - I[2] - I[3]);
    shat[2] = V[2] * conj(I[2]);
    shat[3] = V[3] * conj(I[3] - I[4]);
    shat[4] = V[4] * conj(I[4]);


}
model {
    // standard deviation priors
    sig_e ~ cauchy(0, 1.0);

    // some other priors
    Vreal ~ normal(1.,1.);
    Vimag ~ normal(0.,1.);
    l ~ normal(0.25, 5.);

    // measurements
    for (n in 1:N){
        y_p[n] ~ normal(get_real(shat[n]), sig_e);
        y_q[n] ~ normal(get_imag(shat[n]), sig_e);
        y_mag[n] ~ normal(abs(V[n]), sig_e);
    }

}

generated quantities {
    array[N] real p_hat;
    array[N] real q_hat;
    for (n in 1:N){
        p_hat[n] = get_real(shat[n]);
        q_hat[n] = get_imag(shat[n]);
    }

}