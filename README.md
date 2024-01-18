# Spam_Email_Classification
 Tensorflow Model
The dataset is gotten from https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset and it has columns defined as Columns:
	label
		'1' indicates that the email is classified as spam.
		'0' denotes that the email is legitimate (ham).
	text
		This column contains the actual content of the email messages.

This script utilizes TensorFlow and Pandas to preprocess text data and train a Word2Vec model using the Gensim library on the preprocessed text. Following this, the Python script employs a Keras Tokenizer for text tokenization and constructs a neural network model with an embedding layer, Conv1D, LSTM layers, and Batch Normalization. Notably, the model experienced a substantial improvement in accuracy after the addition of Batch Normalization.

After training, the model is evaluated on a test set, achieving an accuracy of 97%. The script further evaluates the model on custom text inputs before deploying it on Gradio. Using Gradio, the script establishes a straightforward text input interface for real-time model testing. This interface enables users to input text, with the model predicting whether it is spam or not and providing confidence levels for both scenarios. The script concludes by launching the Gradio interface, facilitating interactive testing and sharing.
