import os
os.environ['TESTING'] = '1'

import json
import numpy as np
import pytest

from App import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_home(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b'api prediction de sentiment' in rv.data


def test_predict_post_json_single_string(client, monkeypatch):
    def fake_predict(texts):
        return np.array([[0.8, 0.2]])

    monkeypatch.setattr('App.bert_model.predict', fake_predict, raising=False)

    rv = client.post('/predict', json={'text': 'this is great'})
    assert rv.status_code == 200
    data = rv.get_json()
    assert 'predictions' in data
    assert data['predictions'] == [['POSITIVE', 'NEGATIVE']]


def test_predict_get_query_param(client, monkeypatch):
    def fake_predict(texts):
        return np.array([[0.1, 0.9]])

    monkeypatch.setattr('App.bert_model.predict', fake_predict, raising=False)

    rv = client.get('/predict?text=meh')
    assert rv.status_code == 200
    data = rv.get_json()
    assert data['predictions'] == [['NEGATIVE', 'POSITIVE']]


def test_predict_missing_text_returns_400(client):
    rv = client.post('/predict', json={})
    assert rv.status_code == 400
    data = rv.get_json()
    assert 'error' in data


def test_predict_invalid_format_returns_400(client, monkeypatch):
    # send an object as text which should be invalid
    rv = client.post('/predict', json={'text': {'bad': 'value'}})
    assert rv.status_code == 400
    data = rv.get_json()
    assert 'error' in data
