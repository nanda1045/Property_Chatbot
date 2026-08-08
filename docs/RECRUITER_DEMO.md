# Recruiter Demo

This walkthrough is designed to make the project's agent orchestration, identity boundary, human approval, and operational trace visible in under a minute.

## 30-second identity-security segment

Use local demo mode (`AUTH_MODE=local` and `VITE_AUTH_MODE=local`) with property data loaded and a chat model configured.

1. In **Local demo identity**, choose **Demo Analyst**.
2. Select the **Privileged action** example, or enter:

   > Calculate the average outstanding balance for occupied units with balances above the property's overall average.

3. Submit it. Point out the compact card: **Authorization: DENIED**, current role **Analyst**, required role **PropertyManager**, property scope, and no execution buttons.
4. Open **Run Trace**. Show the authenticated actor, custom-analysis permission, SQL approval request, `sql.approve` denial, property, and completed non-execution path.
5. Switch **Local demo identity** to **Demo Property Manager** and submit the same request again.
6. Show **Authorization: ALLOWED**, expand the validated draft, and explain that the displayed SQL is already stored in the server checkpoint.
7. Click **Approve & Run**. The request sends only the run ID, property, conversation, and decision—not SQL, user ID, or role.
8. Reopen **Run Trace** and show reauthorization, approval granted, approved SQL execution, evidence recorded, evidence verified, and completion.

Takeaway: **The model proposes actions. The application controls authority.**

## What to say if asked

- Local identity selection is a signed, HttpOnly demo mechanism available only when the backend is in local auth mode. It is not a substitute for Entra and resets with the local backend process.
- Entra mode validates the API token and derives object ID, tenant, and app roles from trusted claims. The local role switch endpoint returns `404` in Entra mode.
- Every tool and approval decision is checked against a backend role-permission policy and property grants. Client-supplied role, user, property permission, authorization result, or SQL fields are rejected or stripped.
- Approval rechecks the current actor and property, atomically claims the pending approval, and executes only the validated SQL in the scoped server checkpoint.
- Run Trace is operational telemetry. It records safe identity/permission decisions and execution metadata, never bearer tokens, secrets, hidden prompts, or chain-of-thought.

## Honest limitations

- The local demo signing key is process-local and intentionally ephemeral; after a backend restart, an existing selector fails closed to Demo Viewer until another demo identity is chosen.
- A live custom SQL demo needs loaded MySQL rent-roll data and a configured supported model. Model output must pass the intentionally strict SQL safety validator.
- Standard Entra SPA/API authentication is implemented. This repository does not use or claim Microsoft Entra Agent ID.
