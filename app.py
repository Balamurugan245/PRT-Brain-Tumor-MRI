from flask import Flask, render_template, request
from model import predict
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_feedback(label):
    feedback = {
        "glioma": """
        Glioma Detected

        Gliomas are tumors that originate in the brain's glial cells.
        They can be aggressive and require immediate medical attention.

        Recommended Actions:
        • Consult a neurologist or oncologist immediately  
        • MRI/CT scan confirmation  
        • Possible treatments: surgery, radiation, chemotherapy  

        Early diagnosis improves survival rate significantly.
        """,

        "meningioma": """
        Meningioma Detected

        Meningiomas arise from the meninges (brain covering).
        Most are benign but can grow and affect brain function.

        Recommended Actions:
        • Consult a neurologist  
        • Regular monitoring (if small)  
        • Surgery if tumor grows or causes symptoms  

        Usually manageable with proper medical care.
        """,

        "pituitary": """
        Pituitary Tumor Detected

        Pituitary tumors affect hormone regulation in the body.

        Possible Symptoms:
        • Vision problems  
        • Hormonal imbalance  
        • Headaches  

        Recommended Actions:
        • Endocrinologist consultation  
        • Hormone level testing  
        • MRI scan  

        Most pituitary tumors are treatable.
        """,

        "notumor": """
        No Tumor Detected

        The MRI scan does not show signs of a brain tumor.

        Recommendations:
        • Maintain regular health checkups  
        • Consult doctor if symptoms persist  
        • Follow a healthy lifestyle  

        You are currently in a safe condition.
        """
    }
    return feedback.get(label, "")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        label, confidence = predict(filepath)
        feedback = get_feedback(label)

        return render_template("index.html",
                               prediction=label,
                               confidence=round(confidence*100,2),
                               feedback=feedback,
                               image=filepath)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)