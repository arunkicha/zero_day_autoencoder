# Zero-Day Attack Detection using Autoencoders

This project implements an anomaly detection system using an Autoencoder neural network on the NSL-KDD dataset. The goal is to detect unknown (zero-day) attacks by learning patterns of normal network traffic.

### Approach

The Autoencoder learns a compressed representation of normal behavior. Since it is only exposed to normal data during training:

* Normal samples → reconstructed accurately → low error
* Attack samples → poorly reconstructed → high error

A threshold on reconstruction error is used to classify anomalies.

---

### Key Characteristics

* Training data: **Normal traffic only**
* Learning type: Unsupervised
* Detection method: Reconstruction error thresholding

---

### Performance Behavior

* Clear separation between normal and attack samples
* Higher ROC AUC and precision
* Strong anomaly detection capability

---

### Insight

This setup represents the **ideal scenario for anomaly detection**, where the model learns only normal patterns and treats deviations as anomalies.

---

### Use Case

* Zero-day attack detection
* Intrusion detection systems (IDS)
* Situations where attack data is unavailable or incomplete

---

---

## Conclusion

* Training on **normal-only data** → best for anomaly detection
* Training on **mixed data** → useful for analysis, but degrades performance

This comparison highlights the importance of **training data selection in unsupervised learning systems**.

---

