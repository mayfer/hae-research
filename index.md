# Harmonic Autoencoders: Learning Superposable Representations Through Frequency-Domain Constraints

## Abstract

We propose a novel approach to representation learning using autoencoders that enforces harmonic constraints in the latent space. Traditional sparse autoencoders (SAEs) employ sparsity constraints to learn efficient dictionary representations. Euclidean-space interpretations of these representations often struggle with concept superposition, as vector addition can lead to embeddings that lie outside the manifold of meaningful representations. We propose Harmonic Autoencoders (HAEs), which interpret latent vector indices as frequencies and their values as amplitudes, enabling a fundamentally different approach to representation composition. By incorporating differentiable measures of harmonic and dissonant relationships into the loss function, we encourage the network to learn representations that remain meaningful under superposition, similar to how multiple musical frequencies can combine while preserving their individual characteristics. The model is trained to produce harmonic representations for valid inputs while pushing adversarial or nonsensical inputs towards dissonant regions of the latent space. This approach potentially offers a more natural framework for concept composition and decomposition in neural networks, drawing inspiration from the physics of wave superposition rather than traditional vector space operations.

## Proposed Sections

### Introduction

- Limitations of Euclidean Vector Representations
- Motivation from Wave Superposition
- Related Work in Representation Learning

### Background

- Sparse Autoencoders
- Musical Harmony and Frequency Relationships
- Superposition Principles in Physics and Mathematics

### Harmonic Autoencoder Architecture

- Frequency-Domain Interpretation of Latent Space

## Differentiable Harmony Measures

We propose several differentiable measures for quantifying harmonic relationships in the frequency domain. These measures form the foundation of our training objectives and enable the network to learn representations that preserve meaningful superposition properties.

### 3.1 Frequency Ratio Harmonics

Building on classical music theory, we first define a differentiable measure based on frequency ratios. For two frequencies $f_1$ and $f_2$ with corresponding amplitudes $a_1$ and $a_2$, we define the ratio harmony measure as:

$$H_{\text{ratio}}(f_1, f_2) = -\min_{r} \|\frac{\max(f_1, f_2)}{\min(f_1, f_2)} - r\|_2$$

where $r \in \{1.0, 1.5, 2.0, 2.5, 3.0\}$ represents the set of simple frequency ratios associated with harmonic relationships. The implementation in PyTorch is straightforward:

```python
def ratio_harmony(f1: torch.Tensor, f2: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
    ratio = torch.max(f1, f2) / torch.min(f1, f2)
    simple_ratios = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0], 
                                device=f1.device)
    distances = torch.abs(ratio - simple_ratios)
    # Scale effect by product of amplitudes
    return -torch.min(distances) * a1 * a2
```

### 3.2 Critical Band Theory

A more sophisticated approach draws from psychoacoustics research, particularly the Plomp-Levelt theory of consonance. We define a differentiable dissonance measure based on critical bandwidth:

$$D_{\text{PL}}(f_1, f_2, a_1, a_2) = a_1a_2(e^{-3.5s} - e^{-5.75s})$$

where $s$ is the normalized frequency difference:

$$s = \frac{|f_2 - f_1|}{1.72 \times \frac{(f_1 + f_2)}{2} \times 10^{-3}}$$

The implementation includes critical bandwidth considerations:

```python
def plomp_levelt_dissonance(f1: torch.Tensor, f2: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
    # Constants from Plomp & Levelt's psychoacoustic experiments
    cb = 1.72 * (f1 + f2) / 2 * 0.001  # Critical bandwidth scaling
    s = torch.abs(f2 - f1) / cb        # Normalized frequency difference
    # Dissonance curve with empirically determined decay constants
    d = torch.exp(-3.5 * s) - torch.exp(-5.75 * s)
    return d * a1 * a2
```

### 3.3 Spectral Correlation

For full spectral representations, we introduce a normalized cross-correlation measure:

$$H_{\text{spec}}(s_1, s_2) = \frac{\sum_{i} s_1[i]s_2[i]}{\|s_1\|_2\|s_2\|_2}$$

```python
def spectral_harmony(spec1: torch.Tensor, spec2: torch.Tensor, amp1: torch.Tensor, amp2: torch.Tensor) -> torch.Tensor:
    # Weight spectra by their amplitudes
    weighted_spec1 = spec1 * amp1
    weighted_spec2 = spec2 * amp2
    norm1 = torch.norm(weighted_spec1)
    norm2 = torch.norm(weighted_spec2)
    eps = 1e-8  # Prevent division by zero
    correlation = torch.sum(weighted_spec1 * weighted_spec2) / (norm1 * norm2 + eps)
    return correlation
```

### 3.4 Harmonic Series Alignment

We quantify alignment with the harmonic series through:

$$H_{\text{align}}(F) = -\frac{1}{|F|}\sum_{f \in F}\min_n|f - nf_0|$$

where $F$ is the set of frequencies, $A$ their amplitudes, and $f_0$ the fundamental frequency.

```python
def harmonic_alignment(frequencies: torch.Tensor, amplitudes: torch.Tensor) -> torch.Tensor:
    base_freq = frequencies[0]
    harmonics = torch.arange(1, len(frequencies) + 1, 
                           device=frequencies.device) * base_freq
    # Weight deviations by amplitude
    deviations = torch.min(torch.abs(frequencies.unsqueeze(1) - harmonics), dim=1)[0]
    alignment = -torch.sum(deviations * amplitudes) / (torch.sum(amplitudes) + 1e-8)
    return alignment
```

## Training Objectives and Loss Functions

The overall training objective for our Harmonic Autoencoder combines reconstruction loss with harmonic constraints:


$$\mathcal{L}_\text{total} = \mathcal{L}_\text{recon} + \lambda_h\mathcal{L}_\text{harm} + \lambda_d\mathcal{L}_\text{diss}$$

### 4.1 Combined Harmony Loss

For the harmonic component, we combine multiple measures into a comprehensive harmony loss:

$$\mathcal{L}_\text{harm}(z) = \alpha H_\text{align}(F_z, A_z) + \beta \sum_i \sum_j D_\text{PL}(f_i, f_j, a_i, a_j)$$

where $F_z$ and $A_z$ represent frequencies and amplitudes derived from the latent vector $z$, and $\alpha, \beta$ are weighting parameters.

```python
class HAELoss(nn.Module):
    def __init__(self, lambda_h: float = 1.0, lambda_d: float = 0.5, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.lambda_h = lambda_h
        self.lambda_d = lambda_d
        self.alpha = alpha
        self.beta = beta
        
    def harmony_loss(self, latent_vector: torch.Tensor) -> torch.Tensor:
        frequencies = torch.arange(len(latent_vector), 
                                 device=latent_vector.device)
        amplitudes = latent_vector
        
        # Consider only significant amplitudes
        mask = amplitudes > 1e-6
        active_freqs = frequencies[mask]
        active_amps = amplitudes[mask]
        
        if len(active_freqs) == 0:
            return torch.tensor(0.0, device=latent_vector.device)
        
        # Combine harmony measures
        harm_align = self.alpha * harmonic_alignment(active_freqs, active_amps)
        
        # Compute pairwise dissonance
        n_freqs = len(active_freqs)
        total_dissonance = torch.tensor(0.0, device=latent_vector.device)
        for i in range(n_freqs):
            for j in range(i+1, n_freqs):
                total_dissonance += self.beta * plomp_levelt_dissonance(
                    active_freqs[i], active_freqs[j],
                    active_amps[i], active_amps[j]
                )
        
        return harm_align - total_dissonance
    
    def forward(self, x_recon: torch.Tensor, x: torch.Tensor, z: torch.Tensor, is_valid: torch.Tensor) -> torch.Tensor:
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        
        # Harmony loss (flipped for invalid inputs)
        harm_loss = self.harmony_loss(z)
        harm_loss = torch.where(is_valid, harm_loss, -harm_loss)
        
        return recon_loss + self.lambda_h * harm_loss
```

### 4.2 Training Strategy

During training, we employ a curriculum learning approach where the harmony constraints are gradually increased:

1. Initial phase focuses on reconstruction quality
2. Introduce harmony loss with small $\lambda_h$
3. Gradually increase $\lambda_h$ to target value
4. Fine-tune with adversarial examples to enforce dissonance

This strategy ensures the model first learns to encode and decode effectively before optimizing for harmonic properties.

### Theoretical Framework

- Properties of Harmonic Representations
- Superposition Preservation
- Relationship to Fourier Theory

### Experimental Design

- Datasets and Preprocessing
- Baseline Comparisons
- Evaluation Metrics

### Applications and Future Work

- Concept Composition and Decomposition
- Transfer Learning
- Extensions to Other Domains

### Discussion

- Advantages and Limitations
- Theoretical Implications
- Open Questions