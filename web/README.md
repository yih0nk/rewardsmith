# Frontend draft

Run the dashboard without a backend:

```sh
cd web
npm ci
npm run dev
```

Open the local URL Vite prints (normally http://localhost:5173).

The mock API is enabled by default. It has a small artificial delay and starts
with a completed three-round experiment. Submit a behavior to watch a new run
advance through writing, checking, training, rendering, and critique. Each round
takes about 11 seconds; the full demo takes about 33 seconds. Select a round to
inspect its score, metrics, critique, reward function, and revision diff.

All scores, critiques, and metrics are fixtures, including for custom prompts.
The animated quadruped is an illustrative SVG placeholder, not recorded training
footage. Pause/play controls the illustration. Mock runs are kept in memory and
reset when the page reloads.

To use the backend, create `web/.env.local` containing:

```env
VITE_MOCK_API=0
```

Restart Vite after changing this setting. `src/api.ts` keeps the existing
`/api/requests` endpoints and types; `src/mockApi.ts` implements the corresponding
async calls for the draft. Vite proxies live API and artifact requests to port 8000. Real candidate video paths render as videos when supplied.

```sh
npm run build
```

This checks TypeScript and produces the production bundle in `dist/`.
