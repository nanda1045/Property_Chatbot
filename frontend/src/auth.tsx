import {
  InteractionRequiredAuthError,
  InteractionStatus,
  PublicClientApplication
} from "@azure/msal-browser";
import type { AccountInfo, Configuration } from "@azure/msal-browser";
import { MsalProvider, useMsal } from "@azure/msal-react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState
} from "react";
import type { PropsWithChildren } from "react";

import { getAuthenticatedUser, setAccessTokenProvider } from "./lib/api";
import type { AuthenticatedUser } from "./types";

const AUTH_MODE = (import.meta.env.VITE_AUTH_MODE ?? "local").toLowerCase();
const CLIENT_ID = import.meta.env.VITE_ENTRA_CLIENT_ID ?? "";
const TENANT_ID = import.meta.env.VITE_ENTRA_TENANT_ID ?? "";
const API_SCOPE = import.meta.env.VITE_ENTRA_API_SCOPE ?? "";

type AuthState = {
  mode: "local" | "entra";
  user: AuthenticatedUser | null;
  loading: boolean;
  error: string | null;
  signIn: () => Promise<void>;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<AuthState | null>(null);

const noInteraction = async () => undefined;

function LocalAuthProvider({ children }: PropsWithChildren) {
  const [user, setUser] = useState<AuthenticatedUser | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setAccessTokenProvider(async () => null);
    getAuthenticatedUser()
      .then(setUser)
      .catch((reason: unknown) =>
        setError(reason instanceof Error ? reason.message : "Unable to load local identity.")
      );
  }, []);

  return (
    <AuthContext.Provider
      value={{
        mode: "local",
        user,
        loading: !user && !error,
        error,
        signIn: noInteraction,
        signOut: noInteraction
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

function EntraAuthProvider({ children }: PropsWithChildren) {
  const { instance, accounts, inProgress } = useMsal();
  const account = accounts[0] ?? null;
  const [user, setUser] = useState<AuthenticatedUser | null>(null);
  const [error, setError] = useState<string | null>(null);

  const acquireAccessToken = useCallback(async (): Promise<string | null> => {
    const activeAccount = instance.getActiveAccount() ?? account;
    if (!activeAccount) {
      return null;
    }
    try {
      const response = await instance.acquireTokenSilent({
        account: activeAccount,
        scopes: [API_SCOPE]
      });
      return response.accessToken;
    } catch (reason) {
      if (!(reason instanceof InteractionRequiredAuthError)) {
        throw reason;
      }
      const response = await instance.acquireTokenPopup({
        account: activeAccount,
        scopes: [API_SCOPE]
      });
      return response.accessToken;
    }
  }, [account, instance]);

  useEffect(() => {
    setAccessTokenProvider(acquireAccessToken);
    if (!account) {
      setUser(null);
      return;
    }
    instance.setActiveAccount(account);
    getAuthenticatedUser()
      .then((profile) => {
        setUser(profile);
        setError(null);
      })
      .catch((reason: unknown) => {
        setUser(null);
        setError(reason instanceof Error ? reason.message : "Unable to load identity.");
      });
  }, [account, acquireAccessToken, instance]);

  const signIn = useCallback(async () => {
    setError(null);
    const result = await instance.loginPopup({ scopes: [API_SCOPE] });
    instance.setActiveAccount(result.account);
  }, [instance]);

  const signOut = useCallback(async () => {
    const activeAccount: AccountInfo | null = instance.getActiveAccount() ?? account;
    await instance.logoutPopup({
      account: activeAccount ?? undefined,
      postLogoutRedirectUri: window.location.origin
    });
  }, [account, instance]);

  return (
    <AuthContext.Provider
      value={{
        mode: "entra",
        user,
        loading: inProgress !== InteractionStatus.None || (!!account && !user && !error),
        error,
        signIn,
        signOut
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

function EntraProvider({ children }: PropsWithChildren) {
  const configuration = useMemo<Configuration>(
    () => ({
      auth: {
        clientId: CLIENT_ID,
        authority: `https://login.microsoftonline.com/${TENANT_ID}`,
        redirectUri: window.location.origin,
        postLogoutRedirectUri: window.location.origin
      },
      cache: { cacheLocation: "sessionStorage" }
    }),
    []
  );
  const instance = useMemo(() => new PublicClientApplication(configuration), [configuration]);
  const [initialized, setInitialized] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    instance
      .initialize()
      .then(() => setInitialized(true))
      .catch((reason: unknown) =>
        setError(reason instanceof Error ? reason.message : "MSAL initialization failed.")
      );
  }, [instance]);

  if (error) {
    return <div className="auth-gate"><p className="status-error">{error}</p></div>;
  }
  if (!initialized) {
    return <div className="auth-gate"><p>Initializing Microsoft sign-in…</p></div>;
  }
  return (
    <MsalProvider instance={instance}>
      <EntraAuthProvider>{children}</EntraAuthProvider>
    </MsalProvider>
  );
}

export function AuthProvider({ children }: PropsWithChildren) {
  if (AUTH_MODE === "local") {
    return <LocalAuthProvider>{children}</LocalAuthProvider>;
  }
  if (AUTH_MODE !== "entra" || !CLIENT_ID || !TENANT_ID || !API_SCOPE) {
    return (
      <div className="auth-gate">
        <p className="status-error">Entra authentication is selected but not configured.</p>
      </div>
    );
  }
  return <EntraProvider>{children}</EntraProvider>;
}

export function useAuth(): AuthState {
  const value = useContext(AuthContext);
  if (!value) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return value;
}
