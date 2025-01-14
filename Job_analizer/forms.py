from django import forms

class MyForm(forms.Form):
    miasto = forms.CharField(label='Miasto', max_length=100, required=False)
    zawod = forms.CharField(label='Zawód', max_length=100, required=False)
    #odleglosc_od = forms.CharField(label='Odległość', max_length=100, required=False)
    minimalna_pensja = forms.CharField(label='Minimalna Pensja', max_length=100, required=False)


class MyForm_1(forms.Form):
    miasto_wykres = forms.CharField(label='Miasto', max_length=100, required=False)
    zawod_wykres = forms.CharField(label='Zawód', max_length=100, required=False)

class MyForm_2(forms.Form):
    skrot_kraju = forms.CharField(label='Kraj (skrót)', max_length=100, required=False)

class MyForm_3(forms.Form):
    zawod_wykres_2 = forms.CharField(label='Zawód', max_length=100, required=False)

class MyForm_4(forms.Form):
    miasto_wykres = forms.CharField(label='Miasto', max_length=100, required=False)
    zawod_wykres = forms.CharField(label='Zawód', max_length=100, required=False)

