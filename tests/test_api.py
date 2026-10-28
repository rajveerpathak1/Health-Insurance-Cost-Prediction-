import pytest

import app as app_module


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


VALID = {"age": 40, "sex": "male", "bmi": 31, "children": 2, "smoker": "yes", "region": "southeast"}


def test_home(client):
    assert client.get("/").status_code == 200


def test_predict_ok(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200 and r.get_json()["predicted_cost"] > 0


def test_smoker_costs_more(client):
    s = client.post("/predict", json=VALID).get_json()["predicted_cost"]
    n = client.post("/predict", json={**VALID, "smoker": "no"}).get_json()["predicted_cost"]
    assert s > n


@pytest.mark.parametrize("bad", [{}, {**VALID, "age": "abc"}, {**VALID, "age": 5}, {**VALID, "region": "mars"},
                                 {**VALID, "smoker": "maybe"}])
def test_predict_bad_input(client, bad):
    assert client.post("/predict", json=bad).status_code == 400
