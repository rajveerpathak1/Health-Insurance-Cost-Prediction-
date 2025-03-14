document.getElementById("predictForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    let formData = {
        age: parseInt(document.getElementById("age").value),
        bmi: parseFloat(document.getElementById("bmi").value),
        children: parseInt(document.getElementById("children").value),
        sex: document.getElementById("sex").value.toLowerCase(), // Ensure lowercase
        smoker: document.getElementById("smoker").value.toLowerCase(),
        region: document.getElementById("region").value.toLowerCase()
    };

    try {
        let response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData)
        });

        let result = await response.json();

        if (result.predicted_cost !== undefined) {
            document.getElementById("result").innerText = `Predicted Cost: $${result.predicted_cost}`;
        } else {
            document.getElementById("result").innerText = "Error: " + result.error;
        }
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("result").innerText = "Failed to fetch prediction.";
    }
});
