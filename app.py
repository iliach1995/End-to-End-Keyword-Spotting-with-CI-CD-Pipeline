"""
@author: ilia chiniforooshan

Script to create a web application that wraps the trained model to be used for inference using
`FLASK API`. It facilitates the application to run from a server which defines every routes and
functions to perform. The front-end is designed using `./templates/page.html` and its styles in
`./static/page.css`

Note:
    Make sure to define all the variables and valid paths in `.config_dir/config.yaml` to run
    this script without errors and issues.
"""

from src.inference import KeywordSpotter
from flask import Flask, render_template, request, flash, abort
from config.config_type import DataProcessConfig

app = Flask(__name__)

@app.route('/')
def home():
    """
    Returns the result of calling render_template() with page.html
    """
    return render_template('page.html')

@app.route('/', methods=['GET', 'POST'])
def transcribe():
    """
    Returns the prediction from trained model artifact whenever transcribe route is called.
    It accepts file input (.wav) whenever user uploads the file, and make prediction using it.
    

    Raises
    ------
    NotFoundError: Exception
        404 error, if any exception occurs.
    """
    if request.method == "POST":
        file = request.files['file']

        if file.filename == "":
            flash("File not found !!!", category="error")
            return render_template("page.html")
        
        elif file.filename.endswith('.wav') is False:
            flash("File is not audio .wav !!!", category="error")
            return render_template("page.html")

        else:

            try:
                dataConfig = DataProcessConfig()
                kws = KeywordSpotter("./artifacts/model", dataConfig.n_mfcc, dataConfig.mfcc_length, dataConfig.sampling_rate)

                predicted_keyword , predicted_probability = kws.predict_from_audio(file)
            
            except Exception:
                abort(404, description = "Sorry, something went wrong. Cannot predict from the model. Please try again !!!")
            
            return render_template(
                "page.html",
                 recognized_keyword  = f"Transcribed keyword: {predicted_keyword.title()}",
                 label_probability  = f"Predicted probability: {predicted_probability}"
                 )
        
if __name__ == "__main__":
    app.run(debug=False)