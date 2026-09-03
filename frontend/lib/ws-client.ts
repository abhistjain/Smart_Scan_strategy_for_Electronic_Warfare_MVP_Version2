import type { InitPayload, TickPayload } from "./types";

type Handlers = {
  onInit?: (p: InitPayload) => void;
  onTick?: (p: TickPayload) => void;
  onStatus?: (connected: boolean) => void;
};

/**
 * WebSocket client with automatic exponential-backoff reconnect (spec 11:
 * "WebSocket reconnects gracefully on backend restart").
 */
export class WSClient {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: Handlers;
  private backoff = 500;
  private closedByUser = false;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(scenarioId: string, handlers: Handlers) {
    const httpBase =
      process.env.NEXT_PUBLIC_BACKEND_WS ||
      (typeof window !== "undefined"
        ? `${window.location.protocol === "https:" ? "wss" : "ws"}://127.0.0.1:8000`
        : "ws://127.0.0.1:8000");
    this.url = `${httpBase}/ws/scenario/${scenarioId}`;
    this.handlers = handlers;
  }

  connect() {
    this.closedByUser = false;
    this.open();
  }

  private open() {
    try {
      this.ws = new WebSocket(this.url);
    } catch {
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.backoff = 500;
      this.handlers.onStatus?.(true);
    };

    this.ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (data.type === "init") this.handlers.onInit?.(data as InitPayload);
        else if (data.error) return;
        else this.handlers.onTick?.(data as TickPayload);
      } catch {
        /* ignore malformed frame */
      }
    };

    this.ws.onclose = () => {
      this.handlers.onStatus?.(false);
      if (!this.closedByUser) this.scheduleReconnect();
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => this.open(), this.backoff);
    this.backoff = Math.min(this.backoff * 1.8, 8000);
  }

  send(obj: unknown) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(obj));
    }
  }

  close() {
    this.closedByUser = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
  }
}
