import { Component, input, output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReportTab, ReportTabId } from '../reports-page.state';

@Component({
  selector: 'app-report-tabs',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="card">
      <div class="border-b border-gray-200 dark:border-dark-700">
        <nav class="-mb-px flex space-x-8 px-6">
          @for (tab of tabs(); track tab.id) {
            <button
              (click)="tabChange.emit(tab.id)"
              [class]="tabClass(tab.id)"
              class="py-4 px-1 border-b-2 font-medium text-sm transition-colors"
            >
              <svg
                class="h-5 w-5 mr-2 inline"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  [attr.d]="tab.icon"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                />
              </svg>
              {{ tab.name }}
            </button>
          }
        </nav>
      </div>
    </div>
  `
})
export class ReportTabsComponent {
  tabs = input.required<ReportTab[]>();
  activeTab = input.required<ReportTabId>();
  tabChange = output<ReportTabId>();

  protected tabClass(tabId: ReportTabId): string {
    return tabId === this.activeTab()
      ? 'border-primary-500 text-primary-600'
      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300';
  }
}
