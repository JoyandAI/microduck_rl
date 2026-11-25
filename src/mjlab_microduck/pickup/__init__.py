"""Pick-up detector: a tiny classifier that tells the runtime the duck is being
held by a human (pause the policy) or is back on the floor (resume).

Trained purely in sim: a virtual hand (``hand.py``) picks the robot up while the
deployed policy runs, an oracle pauses/resumes with realistic detection delays,
and the classifier (``model.py``) learns p(held) from a short window of the
signals the runtime reads every 50 Hz tick (``features.py``).
"""
