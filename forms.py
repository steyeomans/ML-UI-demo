from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField, IntegerField, TextAreaField, RadioField, FormField, FieldList, HiddenField, TextField
from wtforms.fields.core import DecimalField
from wtforms.validators import DataRequired, ValidationError, Email, EqualTo, Optional

class LoginForm(FlaskForm):
    email = StringField('email', validators=[DataRequired()])
    password = PasswordField('password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    signin = SubmitField('Sign In')
    newacc = SubmitField('Create New Account')

class RegistrationForm(FlaskForm):
    email = StringField('Email',validators = [DataRequired(), Email()])
    password = PasswordField('Password', validators = [DataRequired()])
    password2 = PasswordField('Confirm Password', validators = [DataRequired(), EqualTo('password', message = 'Passwords must match')])
    submit = SubmitField('Register')

class EditEmailForm(FlaskForm):
    email = StringField('Email', validators = [DataRequired(), Email()])
    submit = SubmitField('Submit')

class EditPwdForm(FlaskForm):
    password = PasswordField('Password', validators = [DataRequired()])
    password2 = PasswordField('Confirm Password', validators = [DataRequired(), EqualTo('password', message = 'Passwords must match')])
    submit = SubmitField('Submit')

class CloseAccForm(FlaskForm):
    password = PasswordField('Password', validators = [DataRequired()])
    submit = SubmitField('Close Account')

class ContactForm(FlaskForm):
    name = StringField('Name', validators = [DataRequired()])
    email = StringField('Email', validators = [DataRequired(), Email()])
    message = TextAreaField('Message', validators = [DataRequired()])
    submit = SubmitField('Submit')

class MLPredForm(FlaskForm):
    med_inc = IntegerField('Median Income')
    avg_house_age = IntegerField('Average House Age')
    avg_rooms = DecimalField('Average Number of Rooms')
    avg_bedrooms = DecimalField('Average Number of Bedrooms')
    population = IntegerField('Population')
    avg_occupancy = DecimalField('Average Occupancy')
    prediction = IntegerField('Median House Value')
    submit = SubmitField('Submit')