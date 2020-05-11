.. _slayerParamsref:

SLAYER Parameter
================

This module provides a way to read the SLAYER configuration parameters from yaml file with dictionary like access. 
A typical yaml configuration file looks like this.

.. code-block:: python
   :linenos:

   simulation:
      Ts: 1.0
      tSample: 1450
   neuron:

   neuron:
      type:     SRMALPHA   # neuron type (available SRMALPHA, LOIHI)
      theta:    10         # neuron threshold
      tauSr:    1.0        # neuron time constant
      tauRef:   1.0        # neuron refractory time constant
      scaleRef: 2          # neuron refractory response scaling (relative to theta)
      tauRho:   1          # spike function derivative time constant (relative to theta)
      scaleRho: 1          # spike function derivative scale factor
   layer:
      - {dim: 34x34x2, wScale: 0.5}
      - {dim: 16c5z}
      - {dim: 2a}
      - {dim: 64c3z}
      - {dim: 2a}
      - {dim: 512}
      - {dim: 10}
   training:
      error:
         type: NumSpikes #ProbSpikes #NumSpikes
         tgtSpikeRegion: {start: 0, stop: 350}
         tgtSpikeCount:  {true: 60, false: 10}
      path:
         out:     Trained/
         in:      path_to_spike_files
         train:   path_to_train_list
         test:    path_to_test_list

For the neuron type ``LOIHI``, more information is provided in :ref:`SLAYER Loihi module <SLAYERLoihimoduleref>`.

.. automodule:: slayerSNN.slayerParams
   :members:
   
