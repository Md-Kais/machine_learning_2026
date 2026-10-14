# ID3 Play Tennis report

The current implementation uses the 14-row Play Tennis example in 15.+Decision+tree+_+ID3.pdf. ID3 chooses multiway categorical splits by base-2 entropy reduction. The algorithm resides in decision_tree_model.py; the training CLI, browser inference and visualization are separate files.

The root entropy is 0.940286. Gains are Outlook 0.246750, Humidity 0.151836, Wind 0.048127 and Temp 0.029223. The learned root is Outlook; Sunny branches split on Humidity, Overcast predicts Yes, and Rain branches split on Wind. For Sunny/Mild/High/Strong, the path is Outlook=Sunny -> Humidity=High -> No.

The webpage displays the full tree, class counts, entropy and gain, and highlights the path used by each prediction. It also shows the original training records and can download the decision graph as SVG. Model fitting uses the lecture records; 100% training accuracy describes reconstruction of the example, not held-out performance.

See README.md for configurable stopping parameters, commands, data format and verification.
