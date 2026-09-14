# A representative likelihood for the 2017 optimization

`likelihood_model.py` exports normalized signal/background histograms to a
Combine datacard and ROOT templates. It is an expected-only analysis development
tool: its observations are the nominal background expectation, and it rejects
observed-data inputs. It fits one region at a time. The result is a reference
against which to test the inexpensive sensitivity ranking, not a reproduction
of the final CMS cross-section result.

The reference is the user-provided **AN-23-067**, particularly §§4.5, 7, 9.1,
9.2, and 10, together with the public source pinned at
[`8ef9b506`](https://github.com/emcannaert/SuuToChiChi-analysis-software/tree/8ef9b506ef4229a2110ec8e662fd5ae93b73420a).
The AN's statistical workspaces/cards are linked to a separate Combine
repository. They are not included in the pinned public analysis source, so the
model does not pretend to copy their undocumented implementation details.

## What the AN actually models

The analysis does not predict the SR through an AT1b/AT0b ABCD ratio. QCD,
ttbar, W+jets, and single-top MC provide the nominal shapes. The dominant QCD
contribution has a shared correction for each group of neighboring mass
superbins:

\[
  b_i(\theta)=q_i\left(1+\beta_{g(i)}\theta_{g(i)}\right)
              +t_i+w_i+u_i,\qquad \theta_g\sim N(0,1),
\]

\[
  \beta_g=\sqrt{\max_{i\in g}\sigma_{\mathrm{BB},i}^2+0.15^2}.
\]

Here `u` denotes single top. The AN equation omits W+jets from its printed
sum, but the sample inventory and systematic discussion include W+jets; the
exporter retains it. The 15% term was motivated by the original analysis's
data/MC comparisons. It is an inherited modeling assumption, and must be
revalidated if reconstruction or selection changes.

The AN fits this model separately in CR, AT1b, and AT0b to validate it. Sharing
nuisance names across separately exported cards would correlate them if someone
later combined the cards; this exporter does not do that combination or assert
that it is justified.

The final mass observable in the AN is a linearized two-dimensional average
superjet mass versus disuperjet mass. Its background-based bin merging targets
a maximum relative statistical error of **17.5%**, equivalent to an effective
count of at least `1 / 0.175**2 = 32.653...`. Groups of neighboring superbins
then share a QCD correction. SR and CR use the SR bin map; AT1b and AT0b use
the AT1b map. A short optimization comparison using fixed coarse Suu mass bins
is an explicitly different, less informative mass model.

## Included priors and missing inputs

The reference configuration copies the numerical normalization uncertainties
given by the AN:

| Nuisance | Processes | 2017 factor |
| --- | --- | --- |
| `xs_QCD` | QCD | `1.30` |
| `xs_TTbar` | ttbar | `1.50` |
| `xs_ST` | single top | `1.30` |
| `xs_WJets` | W+jets | `1.30` |
| `lumi_uncorr17` | All MC, including signal | `1.020` |
| `lumi_corr` | All MC, including signal | `1.009` |

The luminosity components are two correlated-within-2017 nuisance parameters,
not independent errors for each background. The correlated component's name is
also suitable for a future era combination. Cross-section nuisance parameters
are shared across bins of their process. These are log-normal factors: a
single `1.30` entry means multiplication by `1.30` at +1 sigma and division by
`1.30` at -1 sigma.

Experimental and theory shapes cannot be inferred from nominal `sumw2` or by
choosing a plausible percentage. The exporter accepts genuine up/down
histograms and records every missing process/nuisance pair. The AN requires:

* Seven JEC groups: RelativeBal, FlavorQCD, Absolute, BBEC1, and the era-specific
  BBEC1, Absolute, and RelativeSample sources. AK4 and AK8 must vary together,
  with clustering and selection rerun. A supplied `CMS_jec_Total_2017` variation
  is usable for an approximation, but does not satisfy these seven requirements.
* An era-specific JER variation, again varying AK4 and AK8 together.
* Four medium-working-point b-tag shape components: correlated/uncorrelated
  heavy flavor and correlated/uncorrelated light flavor.
* Pileup and L1 prefiring weight variations. The AN labels prefiring a
  normalization uncertainty but describes varying event weights; supplied
  binned templates retain any resulting shape dependence.
* PDF variations for QCD and signal separately, with a shared PDF nuisance for
  ttbar and W+jets. The note uses an event-level replica treatment for QCD and
  a histogram-level Hessian construction for the other listed samples.
* Process scale variations. The pT-binned QCD samples lack the stored scale
  weights, so the AN obtains them from HT-binned QCD samples. Single-top PDF
  and scale effects use normalization nuisances in the AN, but numerical
  amplitudes are not supplied in its text and are deliberately left missing.

Names in `required_shape_nuisances` are the local interface, following the
AN's documented correlations. They are not a claim that the private cards use
every exact spelling. Additional real variations, such as a one-sided top-pT
variation from the processor, can be supplied with a nominal down template.
The exporter does not invent the opposite variation or apply an arbitrary
envelope. It also does not silently apply the AN's neighbor smoothing and
manual symmetrization. Such processing requires the actual bin-neighbor map
and a separate audit of the shifted histograms.

## Statistical treatment and its scope

Each input mass bin becomes a one-bin Combine channel. A QCD group has one
Gaussian nuisance and one linear `rateParam` factor in every affected channel.
This exactly represents the displayed linear group factor without requiring
the private RooParametricHist workspace code. Supplied shape templates share
the same nuisance across bins and processes. Their ±1 sigma yields are
preserved, but Combine's normalization interpolation in the one-bin channels
is not identical to a morph of the original multibin shape between those
endpoints. The standard card conventions are described in the
[Combine datacard documentation](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part2/settinguptheanalysis/).

There are two supported finite-MC treatments:

1. By default, each QCD group's coefficient uses the maximum **QCD-only**
   `sqrt(sumw2)/sumw` in the group, in quadrature with 15%. Every non-QCD
   process/bin and signal/bin has its own Gaussian statistical factor with
   width `sqrt(sumw2)/sumw`. Non-QCD variances are preserved. The shared QCD
   factor replaces independent-bin MC covariance with a fully correlated
   fluctuation within each group, using its largest relative error; this can
   inflate the variance of better populated bins. It is an AN-inspired
   approximation, not a verified implementation of the AN's Beeston–Barlow term.
2. An audited `bb_relative_uncertainty` array can be supplied along with
   `bb_uncertainty_provenance`. The QCD coefficient then uses its group maximum,
   and no second non-QCD background MC nuisance is added because the supplied
   coefficients already absorb background statistics. Signal MC remains
   separate. A bin with no QCD cannot absorb background statistics and is
   rejected in this mode if any background variance remains.

The `statistics_only` stage removes the analysis normalization/shape priors
and QCD blanket term while retaining exactly the same grouped MC basis as
`reference`. Both stages use the configured groups, the same maximum relative
MC error within each group, and the same independent non-QCD/signal factors.
If audited aggregate BB coefficients are supplied, both stages use them and
both omit the already absorbed background MC factors. Thus reference broadens
each QCD coefficient from `delta_MC` to `sqrt(delta_MC² + 0.15²)` and adds the
implemented analysis priors without removing independent fluctuation modes
from the baseline. Setting the blanket term to zero and removing analysis
priors produces identical datacards in both stages.

The MC contribution and blanket contribution use one combined linear Gaussian
factor per group, not separately fitted nuisance parameters. In the untruncated
Gaussian approximation this gives the same covariance along that group
direction as adding a closure variance; the finite positive-yield domains mean
it should not be described as an exact equivalence between independently
truncated sources. Groups are independent of one another. The Gaussian MC
constraints also differ from the effective-Poisson auxiliary treatment in the
fast analytic score.

`stage_contract_version: 2` and `finite_mc_basis_sha256` in `model.json` identify
this controlled comparison. The common basis and its widths are recorded in
`finite_mc_basis`. Earlier manifests without this version used independent-bin
QCD factors in `statistics_only` and grouped factors in `reference`. Those
stages compared different covariance assumptions: reference could improve
sensitivity by removing bin-to-bin fluctuation modes, so their difference does
not isolate the effect of analysis uncertainties. Previously generated cards
and fit results remain unchanged; regenerate both stages in a new output
directory to use version 2.

There is **no autoMCStats line** in the card. ROOT bin errors preserve the
input `sumw2` for inspection, but the statistical nuisances are all explicit;
QCD MC is never counted twice.

Linear factors cannot have negative yields. Each nuisance has the recorded
domain `max(-8, -(1-1e-8)/relative_width) < theta < 8`. The exporter sets the
lower endpoint a small distance inside the positive domain; it does not clip
the yield function during fitting. Large relative uncertainties can truncate
the Gaussian within five sigma, which is flagged in `model.json`. A fit near
this boundary needs additional validation or better MC, and asymptotic
coverage is not guaranteed merely because Combine returned a number.

The default MC guard uses the AN's 17.5% target in every signal-populated bin.
`--allow-unsupported-mc` permits a **diagnostic** card with the failed bins
recorded; it does not turn that candidate into a supported ranking result.
Negative component yields, signed cancellation to zero with nonzero variance,
and signal with zero background are always rejected. There is no epsilon
background insertion. A variation that migrates into a nominally empty process
bin, or removes its yield entirely, requires a common bin merging decision
before export. The builder cannot hide that problem with template flooring.

## Input interface and use

Input arrays are already normalized expected yields. One signal mass/decay
hypothesis is exported at a time:

```json
{
  "schema_version": 1,
  "provenance": {
    "signal": "WbWb_6000_2000",
    "signal_cross_section_pb": 0.005711471665638484,
    "input_kind": "evaluated_ntuples"
  },
  "channels": {
    "SR": {
      "bin_edges": [0, 3500, 5500, 7500, 12000],
      "qcd_groups": [[0], [1], [2], [3]],
      "processes": {
        "signal": {"sumw": [0, 3, 20, 1], "sumw2": [0, 0.1, 0.2, 0.01]},
        "QCD": {
          "sumw": [100, 100, 100, 100],
          "sumw2": [10, 10, 10, 10],
          "variations": {
            "CMS_pu": {"up": [102, 101, 99, 103], "down": [98, 99, 101, 97]}
          }
        }
      }
    }
  }
}
```

These numbers illustrate the schema only; they are not MC results. Supported
process names are `signal`, `QCD`, `TTbar`, `ST`, and `WJets`. Absent background
processes are represented by zeros. Two-dimensional arrays must first be
flattened, with `original_shape` and `bin_axes_gev` preserving their meaning.
The `from_objective()` adapter does this in C order and aggregates campaign
sample categories while retaining variances and provenance. It checks
campaign/objective normalization consistency when that metadata is available.

```bash
python3 likelihood_model.py \
  --objective run/evaluation/objective.json \
  --campaign config/sensitivity_2017.json \
  --signal WbWb_6000_2000 \
  --output-dir run/likelihood/WbWb_6000_2000

python3 likelihood_model.py \
  --input reference_histograms.json --region CR \
  --output-dir run/likelihood/CR
```

The Python interface is
`export_model(inputs, output_dir, config=None, region="SR", stage="reference")`.
It creates `datacard.txt`, `templates.root`, `inputs.json`, `model_config.json`,
and `model.json`, and refuses to overwrite an existing model. Hashes of the
inputs, configuration, and emitted artifacts tie every result to its actual
templates. `model.json` lists missing nuisances, missing reference audits,
low-MC bins, truncated nuisance domains, and modeling approximations.

`--require-complete-reference` fails unless the required templates,
normalizations, audited BB coefficients, MC support, and provenance audits for
selection, normalization, templates, bin mapping, and control-region closure
are all supplied. Even complete inputs do not certify an official CMS result:
`full_an_reproduction` and `production_ready` remain false in this development
exporter.

The parameter `r` scales the specified signal template. Multiplying the median
expected `r` limit by `provenance.signal_cross_section_pb` gives a limit in the
**same cross-section convention as that template**. The current benchmark
catalog includes the selected fully hadronic WbWb decay factors. This is not
automatically an inclusive Suu production cross-section limit; converting to
one requires the corresponding decay and branching assumptions.

## Validation

`test/test_likelihood_model.py` verifies shared QCD group widths, positive
nuisance domains, independent non-QCD and signal statistical errors, the
audited-BB alternative, missing-template guards, region isolation, signed/empty
bin rejection, exact ROOT template contents and errors, and preservation of
normalization and variances in the objective adapter. A real Combine runtime
check should additionally inspect the generated workspace, expected limits,
discovery significance, and fit diagnostics; a successfully written datacard
alone is not evidence that the likelihood is healthy.
