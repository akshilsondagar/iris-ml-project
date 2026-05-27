from django.shortcuts import render
from .forms import IrisForm
from .ml_model import model, flower_names, accuracy

def predict_iris(request):

    prediction = None

    if request.method == 'POST':

        form = IrisForm(request.POST)

        if form.is_valid():

            sepal_length = form.cleaned_data['sepal_length']
            sepal_width = form.cleaned_data['sepal_width']
            petal_length = form.cleaned_data['petal_length']
            petal_width = form.cleaned_data['petal_width']

            features = [[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]]

            result = model.predict(features)[0]

            prediction = flower_names[result]

    else:
        form = IrisForm()

    return render(request, 'predictor/index.html', {
        'form': form,
        'prediction': prediction,
        'accuracy': round(accuracy * 100, 2)
    })