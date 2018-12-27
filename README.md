# CESLeA_speaker_recognition
This code implements speaker recognition using hmmlearn which is the python library for implementing HMM. In this implementation Gaussian HMM is used i.e. each state of HMM is represented by a single gaussian. This will result in using greater number of HMM states to model a speaker as compared to when GMM (Gaussian Mixture Models) is used to represent each state of HMM. 

This is a complete code but changes are still been made to improve this code.

'utils.py' provides some useful parameters to be used in the  main code file i.e. speaker_recog.py.

"hmm_spr.pkl" is the file in which speaker HMM models are saved for later use without training again.
