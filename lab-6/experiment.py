"""Real local protocol computations; the viewer replays only recorded events.

The process is a teaching simulation, not a security sandbox. The HE server
boundary is nevertheless real: only serialized public context/ciphertexts enter.
Salary and prediction errors are in thousands of dollars ($k).
"""
from __future__ import annotations

import base64
import inspect
from io import BytesIO
from time import perf_counter

import numpy as np
import tenseal as ts

HE_STARTER = '''def client_encrypt(private, features):
    # CLIENT: use ts.ckks_vector and return serialized ciphertext bytes.
    raise NotImplementedError("Complete client encryption")

def encrypted_predict(encrypted_x, weights, bias):
    # SERVER: return encrypted_x dot weights plus bias; do not decrypt.
    raise NotImplementedError("Complete encrypted prediction")

def client_decrypt(private, prediction_bytes):
    # CLIENT: reconstruct the encrypted reply with the private context,
    # decrypt it, and return the first (and only) prediction.
    raise NotImplementedError("Complete client decryption")
'''
FL_STARTER = '''def client_gradient(theta, local_X, local_y):
    # CLIENT: gradient of mean squared error. X includes a column of ones.
    raise NotImplementedError("Complete client gradient")

def aggregate_gradients(gradients, counts):
    # SERVER: sample-count-weighted mean of the gradient vectors.
    raise NotImplementedError("Complete gradient aggregation")

def server_rounds(theta, num_rounds, learning_rate, send_to_clients):
    # SERVER: repeat num_rounds times. The provided communication adapter
    # returns (gradients, counts) from clients using your client_gradient.
    # Average gradients, update theta, then yield theta.copy() each round.
    raise NotImplementedError("Complete server rounds")
'''
WEIGHTS = np.array([12.0, 8.0])
BIAS = 35.0
INPUTS = [[0.2, 0.25], [0.5, 0.75], [0.9, 1.0]]
SIZES = [20, 60, 120]
LR = 0.1
LOCAL_STEPS = 5


def client_create_keys():
    """CLIENT: keep the private context; export no secret key."""
    private = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                         coeff_mod_bit_sizes=[60, 40, 40, 60], n_threads=1)
    private.global_scale = 2**40
    private.generate_galois_keys()
    public_bytes = private.serialize(save_secret_key=False)
    return private, public_bytes


def client_encrypt(private, features):
    """CLIENT: only these ciphertext bytes cross the boundary."""
    return ts.ckks_vector(private, features).serialize()


def server_receive(public_bytes, ciphertext_bytes):
    """SERVER: reconstruct using a context without the secret key."""
    public = ts.context_from(public_bytes, n_threads=1)
    assert not public.has_secret_key()
    encrypted_x = ts.ckks_vector_from(public, ciphertext_bytes)
    return public, encrypted_x


def client_decrypt(private, prediction_bytes):
    """CLIENT: the server's reply is decrypted only here."""
    return ts.ckks_vector_from(private, prediction_bytes).decrypt()[0]


def client_train(theta, local_X, local_y, steps=LOCAL_STEPS):
    """CLIENT: raw records are local; send parameters and count only."""
    updated = theta.copy()
    for _ in range(steps):
        residual = local_X @ updated - local_y
        gradient = 2 * local_X.T @ residual / len(local_y)
        updated -= LR * gradient
    local_loss = float(np.mean((local_X @ updated - local_y)**2))
    return updated, len(local_y), local_loss


def server_broadcast(theta):
    """SERVER: every client starts the round from this same vector."""
    return [theta.copy() for _ in SIZES]


def instructor_evaluate(theta, test_X, test_y):
    """INSTRUCTOR EVALUATION: held-out examples never enter messages."""
    return float(np.mean((test_X @ theta - test_y)**2))


def compile_student(source, name):
    # Student Python has ordinary local Python access, as in earlier labs.
    namespace = {"np": np, "ts": ts}
    exec(compile(source, f"<student {name}>", "exec"), namespace)
    fn = namespace.get(name)
    if not callable(fn):
        raise ValueError(f"Define a function named {name}")
    return fn


def event(events, actor, code, text, updates=None, message=None, metric=None):
    events.append(dict(actor=actor, code=code, text=text, updates=updates or {},
                       message=message, metric=metric))


def message(sender, recipient, kind, payload):
    return dict(sender=sender, recipient=recipient, kind=kind, payload=payload)


def byte_info(data):
    return {"bytes": len(data), "hex_preview": data[:18].hex() + "…"}


def run_he(source, inputs=INPUTS, weights=WEIGHTS, bias=BIAS):
    events, rows = [], []
    result = dict(events=events, rows=rows, error=None, checks={}, elapsed_ms=0)
    start = perf_counter()
    stage, actor, code = "Compile server function", "server", "he_server"
    try:
        predict = compile_student(source, "encrypted_predict")
        encrypt = compile_student(source, "client_encrypt")
        decrypt = compile_student(source, "client_decrypt")
        stage, actor, code = "Create client keys", "client", "keys"
        private, public_bytes = client_create_keys()
        event(events, actor, code, "Client creates keys. Secret key stays here.",
              {"client": {"secret_key": "LOCAL ONLY", "inputs": inputs}})
        event(events, "client", "keys", "Send public/evaluation context; no secret key.",
              {"server": {"context": "Public + evaluation keys only"}},
              message("client", "server", "Public/evaluation context", byte_info(public_bytes)))
        for index, features in enumerate(inputs, 1):
            stage, actor, code = f"Encrypt input {index}", "client", "encrypt"
            t = perf_counter()
            ciphertext_bytes = encrypt(private, features)
            if not isinstance(ciphertext_bytes, bytes):
                raise TypeError("client_encrypt must return serialized ciphertext bytes")
            encrypt_ms = (perf_counter() - t) * 1000
            event(events, actor, code, f"Client encrypts input {index}.",
                  {"client": {"current_features": features, "ciphertext_bytes": len(ciphertext_bytes)}})
            stage, actor, code = f"Receive input {index}", "server", "receive"
            public, encrypted_x = server_receive(public_bytes, ciphertext_bytes)
            event(events, "server", code, f"Server receives encrypted input {index}.",
                  {"server": {"input": byte_info(ciphertext_bytes), "secret_key": "ABSENT"}},
                  message("client", "server", "Encrypted feature vector", byte_info(ciphertext_bytes)))
            stage, actor, code = f"Predict input {index}", "server", "he_server"
            t = perf_counter()
            encrypted_y = predict(encrypted_x, np.asarray(weights).tolist(), float(bias))
            if not isinstance(encrypted_y, ts.CKKSVector):
                raise TypeError("Return a CKKSVector containing one encrypted prediction")
            if encrypted_y.size() != 1:
                raise ValueError("Prediction must contain exactly one encrypted scalar")
            prediction_bytes = encrypted_y.serialize()
            predict_ms = (perf_counter() - t) * 1000
            try:
                encrypted_x.decrypt()
            except (ValueError, RuntimeError):
                decrypt_blocked = True
            else:
                decrypt_blocked = False
            event(events, actor, code, "Server computes using ciphertext and plaintext model weights.",
                  {"server": {"prediction": "Encrypted scalar", "computation_ms": round(predict_ms, 3)}})
            event(events, "server", "he_server", "Return encrypted prediction to client.",
                  {"client": {"reply": byte_info(prediction_bytes)}},
                  message("server", "client", "Encrypted prediction", byte_info(prediction_bytes)))
            stage, actor, code = f"Decrypt input {index}", "client", "decrypt"
            t = perf_counter()
            prediction = float(decrypt(private, prediction_bytes))
            if not np.isfinite(prediction):
                raise ValueError("client_decrypt must return a finite numeric prediction")
            decrypt_ms = (perf_counter() - t) * 1000
            event(events, actor, code, "Client decrypts the salary prediction ($k).",
                  {"client": {"prediction_$k": prediction}})
            stage, actor, code = f"Evaluate input {index}", "instructor", "he_eval"
            t = perf_counter()
            reference = float(np.dot(features, weights) + bias)
            plain_us = (perf_counter() - t) * 1e6
            row = dict(input=index, features=features, prediction=prediction,
                       reference=reference, error=abs(prediction-reference),
                       encrypt_ms=encrypt_ms, server_ms=predict_ms, decrypt_ms=decrypt_ms,
                       plaintext_us=plain_us)
            rows.append(row)
            event(events, actor, code, "Instructor compares with a separate plaintext reference.",
                  {"instructor": row})
        result["checks"] = {"Server has no secret key": not public.has_secret_key(),
                            "Server decryption is blocked": decrypt_blocked,
                            "All three errors ≤ 0.0001 $k": len(rows) == 3 and all(r["error"] <= 1e-4 for r in rows)}
    except Exception as exc:
        result["error"] = f"{stage}: {type(exc).__name__}: {exc}"
        event(events, actor, code, result["error"], {actor: {"error": result["error"]}})
    result["elapsed_ms"] = (perf_counter() - start) * 1000
    return result


def dataset():
    """Distinct synthetic employees; years are converted to model feature units."""
    rng = np.random.default_rng(302)
    def make(n, company_offset=0):
        experience = np.clip(np.round(rng.gamma(2.4, 3.0, n), 1), 0, 20)
        education = rng.choice([0, 1, 2, 3, 4], n, p=[.12, .08, .25, .15, .40])
        features = np.column_stack([experience / 20, education / 4])
        X = np.column_stack([features, np.ones(n)])
        y = features @ WEIGHTS + BIAS + company_offset + rng.normal(0, 1.5, n)
        return X, y
    return [make(n, offset) for n, offset in zip(SIZES, [-1., .5, 1.])], make(100)


def baseline_models(clients, test):
    def train(X, y):
        return client_train(np.zeros(3), X, y, steps=100)[0]
    pooled_X = np.concatenate([X for X, _ in clients])
    pooled_y = np.concatenate([y for _, y in clients])
    return {"Centralized": instructor_evaluate(train(pooled_X, pooled_y), *test),
            **{f"Client {i+1} only": instructor_evaluate(train(X, y), *test)
               for i, (X, y) in enumerate(clients)}}


def fl_settings(num_rounds=1, learning_rate=LR):
    rounds, rate = num_rounds, learning_rate
    if not isinstance(rounds, int) or isinstance(rounds, bool) or not 1 <= rounds <= 200:
        raise ValueError("Training rounds must be an integer from 1 to 200")
    if not np.isscalar(rate) or not np.isfinite(rate) or rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    return rounds, float(rate)


def run_fl(source, target_round=None, previous=None, num_rounds=1, learning_rate=LR):
    events = []
    clients, test = dataset()
    theta = np.array(previous["theta"], dtype=float) if previous else np.zeros(3)
    history = list(previous["history"]) if previous else [dict(round=0, mse=instructor_evaluate(theta, *test))]
    result = dict(events=events, history=history, theta=theta.tolist(), error=None,
                  checks={}, baselines={}, elapsed_ms=0)
    start = perf_counter()
    stage, actor, code = "Compile student functions", "server", "fl_server"
    try:
        rounds, rate = fl_settings(num_rounds, learning_rate)
        target = rounds if target_round is None else min(rounds, target_round)
        result.update(num_rounds=rounds, learning_rate=rate)
        loop = compile_student(source, "server_rounds")
        gradient_fn = loop.__globals__.get("client_gradient")
        if not callable(gradient_fn) or not callable(loop.__globals__.get("aggregate_gradients")):
            raise ValueError("Define client_gradient and aggregate_gradients")
        def train_reference(X, y):
            value = np.zeros(3)
            for _ in range(rounds):
                value -= rate * (2 * X.T @ (X @ value-y) / len(y))
            return instructor_evaluate(value, *test)
        result["baselines"] = {"Centralized": train_reference(np.vstack([X for X,y in clients]), np.concatenate([y for X,y in clients])),
            **{f"Client {i+1} only": train_reference(X,y) for i,(X,y) in enumerate(clients)}}
        for i,(X,y) in enumerate(clients,1):
            event(events,f"c{i}","train","Training records remain at this company.",
                  {f"c{i}":{"samples":len(y),"record_preview":[{"experience_years":round(float(row[0]*20),1),"education_years":round(float(row[1]*4),1),"salary_$k":round(float(salary),2)} for row,salary in zip(X[:5],y[:5])],"records":"LOCAL ONLY"}})
        initial = history[-1]["round"]
        calls = 0
        last = None
        checks = {"Client gradients match MSE derivatives": True,
                  "Server update matches weighted FedSGD": True,
                  "Same model sent to every client": True}
        def send_to_clients(current):
            nonlocal stage, actor, code, calls, last
            stage,actor,code = "Server broadcast", "server", "fl_server"
            current = np.asarray(current,dtype=float)
            if current.shape != (3,) or not np.all(np.isfinite(current)):
                raise ValueError("Broadcast a finite model vector with shape (3,)")
            calls += 1
            if calls > target-initial:
                raise ValueError("Call send_to_clients exactly once per requested round")
            rnd=initial+calls
            starts=server_broadcast(current)
            gradients=[];counts=[];client_states={}
            detailed=rnd==1 or target==initial+1
            for i,(parameters,(X,y)) in enumerate(zip(starts,clients),1):
                if detailed:
                    event(events,"server","fl_server",f"Round {rnd}: broadcast the same model to company {i}.",
                        {f"c{i}":{"starting_parameters":parameters.tolist(),"round":rnd,"gradient":None,"local_mse":None}},
                        message("server",f"c{i}","Global model parameters",{"parameters":parameters.tolist(),"round":rnd}))
                stage,actor,code=f"Round {rnd}: company {i} client gradient",f"c{i}","train"
                snapshot=parameters.copy()
                grad=np.asarray(gradient_fn(parameters,X.copy(),y.copy()),dtype=float)
                if grad.shape!=(3,) or not np.all(np.isfinite(grad)):
                    raise ValueError("client_gradient must return a finite vector with shape (3,)")
                reference=2*X.T@(X@snapshot-y)/len(y)
                checks["Client gradients match MSE derivatives"] &= bool(np.allclose(grad,reference))
                checks["Same model sent to every client"] &= bool(np.array_equal(snapshot,current))
                gradients.append(grad.copy());counts.append(len(y))
                client_states[f"c{i}"] = {"round":rnd,"starting_parameters":snapshot.tolist(),
                    "gradient":grad.tolist(),"local_mse":float(np.mean((X@snapshot-y)**2))}
                if detailed:
                    event(events,actor,"train",f"Company {i} computes one gradient at the shared model.",
                        {actor:client_states[actor]})
                    event(events,"server","fl_server",f"Company {i} sends gradient and sample count.",
                        {"server":{f"received_c{i}":{"gradient":grad.tolist(),"sample_count":len(y)}}},
                        message(actor,"server","Gradient + sample count",{"gradient":grad.tolist(),"sample_count":len(y)}))
            last=(current.copy(),[g.copy() for g in gradients],counts.copy(),detailed,client_states)
            stage,actor,code=f"Round {rnd}: server aggregation and update","server","fl_server"
            return gradients,counts
        outputs=loop(theta.copy(),target-initial,rate,send_to_clients)
        if outputs is None:
            raise TypeError("server_rounds must yield the updated model after every round")
        for value in outputs:
            stage,actor,code="Server round result","server","fl_server"
            if last is None or calls!=len(history)-initial:
                raise ValueError("Yield exactly once after each send_to_clients call")
            actual=np.asarray(value,dtype=float)
            if actual.shape!=(3,) or not np.all(np.isfinite(actual)):
                raise ValueError("Yield a finite model vector with shape (3,)")
            before,gs,counts,detailed,client_states=last
            expected=before-rate*np.average(gs,axis=0,weights=counts)
            checks["Server update matches weighted FedSGD"] &= bool(np.allclose(actual,expected))
            checks["Same model sent to every client"] &= bool(np.array_equal(before,theta))
            theta=actual.copy();result["theta"]=theta.tolist()
            rnd=initial+calls
            if detailed:
                event(events,"server","fl_server","Server applies your weighted aggregation and round update (actual result).",
                    {"server":{"parameters":theta.tolist(),"weighted_gradient_reference":np.average(gs,axis=0,weights=counts).tolist()}})
            else:
                result.setdefault("round_summaries",[]).append({"round":rnd,"broadcast":{"parameters":before.tolist()},
                    "uploads":[{"sender":f"c{i}","gradient":g.tolist(),"sample_count":n} for i,(g,n) in enumerate(zip(gs,counts),1)]})
                # Compact playback still refreshes every client using this
                # round's actual local computation, after its simulated messages.
                summary_updates = {**client_states, "server": {
                    "round":rnd,"parameters":theta.tolist(),
                    "weighted_gradient_reference":np.average(gs,axis=0,weights=counts).tolist(),
                    **{f"received_c{i}":{"gradient":g.tolist(),"sample_count":n}
                       for i,(g,n) in enumerate(zip(gs,counts),1)}}}
                event(events,"server","fl_server",
                    f"Round {rnd}: broadcast, client gradients, uploads, and server update complete.",
                    summary_updates)
            metric=dict(round=rnd,mse=instructor_evaluate(theta,*test));history.append(metric)
            event(events,"instructor","fl_eval",f"Round {rnd}: evaluate on separate held-out records.",
                {"instructor":metric,"server":{"round":rnd,"parameters":theta.tolist()}},metric=metric)
            last=None
        if calls!=target-initial or last is not None:
            raise ValueError("server_rounds must execute and yield exactly num_rounds updates")
        checks.update({"Training goal: held-out MSE ≤ 3.0 $k²":history[-1]["mse"]<=3.0,"Only gradients and counts uploaded":True,"Test error below initialization":history[-1]["mse"]<history[0]["mse"]})
        result["checks"]={k:bool(v) for k,v in checks.items()}
    except Exception as exc:
        result["error"]=f"{stage}: {type(exc).__name__}: {exc}"
        event(events,actor,code,result["error"],{actor:{"error":result["error"]}})
    result["elapsed_ms"]=(perf_counter()-start)*1000
    return result


def sources():
    return {"keys": inspect.getsource(client_create_keys), "encrypt": inspect.getsource(client_encrypt),
            "receive": inspect.getsource(server_receive), "decrypt": inspect.getsource(client_decrypt),
            "train": "# YOUR CLIENT CODE: client_gradient(theta, local_X, local_y)", "broadcast": inspect.getsource(server_broadcast),
            "he_eval": "# INSTRUCTOR EVALUATION — separate from the server\nreference = float(np.dot(features, weights) + bias)",
            "fl_eval": inspect.getsource(instructor_evaluate)}


def submission_card(he, fl, linked):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor("#f7f9fc")
    fig.suptitle("Lab 6 · Federated Learning & Homomorphic Encryption", fontsize=18, fontweight="bold")
    axes[0].axis("off")
    lines = ["HE: plaintext vs decrypted ($k)"]
    for row in he.get("rows", []):
        lines.append(f"Input {row['input']}: {row['reference']:.4f} / {row['prediction']:.4f}  |Δ|={row['error']:.2g}")
    lines += ["", f"FL: completed round {fl['history'][-1]['round']}"]
    checks = {**he.get("checks", {}), **fl.get("checks", {})}
    if linked:
        checks["Federated model: encrypted prediction agreement"] = bool(linked.get("checks", {}).get("All three errors ≤ 0.0001 $k", False))
    for name, ok in checks.items():
        lines.append(f"{'PASS' if ok else 'FAIL'} · {name}")
    if he.get("error") or fl.get("error") or (linked and linked.get("error")):
        lines.append("INCOMPLETE: resolve execution errors")
    axes[0].text(0, 1, "\n".join(lines), va="top", fontsize=9, linespacing=1.65)
    axes[1].plot([r["round"] for r in fl["history"]], [r["mse"] for r in fl["history"]], "o-", label="Federated")
    for (label, mse), color in zip(fl["baselines"].items(), ["#637184", "#aa7340", "#4d7894", "#9d6696"]):
        axes[1].axhline(mse, ls="--", lw=1, label=label, color=color)
    axes[1].set(xlabel="Communication round", ylabel="Held-out MSE ($k²)", yscale="log", title="Instructor evaluation")
    axes[1].legend(fontsize=8)
    fig.subplots_adjust(top=.84, bottom=.12, left=.05, right=.97, wspace=.2)
    stream = BytesIO()
    fig.savefig(stream, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(stream.getvalue()).decode("ascii")
