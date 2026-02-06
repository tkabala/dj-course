# MCP Playwright

Projekt demonstracyjny integracji Claude Code z Playwright poprzez MCP (Model Context Protocol). Umozliwia sterowanie przegladarka z poziomu Claude Code - nawigowanie po stronach, klikanie elementow, wypelnianie formularzy i robienie screenshotow.

## Wymagania

- Node.js
- Przegladarki Playwright (`npx playwright install`)

## Instalacja

```bash
npm install
npx playwright install
```

## Konfiguracja MCP

Plik `.mcp.json` konfiguruje serwer Playwright MCP dla Claude Code:

```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["@playwright/mcp@latest"]
    }
  }
}
```

## Testy E2E

Testy znajduja sie w katalogu `tests/`:

- `hello.spec.ts` - podstawowy test weryfikujacy tytul strony Playwright
- `tms-order-wizard.spec.ts` - testy aplikacji TMS (rejestracja, logowanie, tworzenie zamowienia)

### Uruchamianie testow

```bash
# Wszystkie testy
npx playwright test

# Konkretny plik
npx playwright test tests/hello.spec.ts

# Z widoczna przegladarka
npx playwright test --headed

# Tryb debug
npx playwright test --debug

# Raport z testow
npx playwright show-report
```

## Przydatne komendy MCP (w Claude Code)

W trakcie rozmowy z Claude Code dostepne sa narzedzia Playwright MCP:

| Komenda | Opis |
|---------|------|
| `browser_navigate` | Przejdz pod URL |
| `browser_snapshot` | Pobierz snapshot dostepnosci strony |
| `browser_click` | Kliknij element |
| `browser_type` | Wpisz tekst w pole |
| `browser_fill_form` | Wypelnij formularz |
| `browser_take_screenshot` | Zrob screenshot |
| `browser_press_key` | Wcisnij klawisz |
| `browser_select_option` | Wybierz opcje z dropdown |
| `browser_console_messages` | Pokaz logi konsoli |
| `browser_network_requests` | Pokaz zapytania sieciowe |
| `browser_tabs` | Zarzadzanie zakladkami |
| `browser_close` | Zamknij przegladarke |

## Struktura projektu

```
.mcp.json              # Konfiguracja serwera MCP
package.json           # Zaleznosci (Playwright)
playwright.config.ts   # Konfiguracja testow Playwright
tests/                 # Testy E2E
```
