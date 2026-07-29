import { Component, input, output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DropdownComponent } from '../../../ui-library/Dropdown.component';
import { Heading3Component } from '../../../ui-library/Typography/Typography.component';
import { AuditTrail } from '../reports.model';
import { KpiCardComponent } from './kpi-card.component';

@Component({
  selector: 'app-audit-trails',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    DropdownComponent,
    Heading3Component,
    KpiCardComponent
  ],
  template: `
    <div class="p-6">
      <div class="flex flex-col space-y-4 mb-6">
        <div class="flex justify-between items-center">
          <ui-heading3>Audit Trails</ui-heading3>
          <button (click)="export.emit()" class="btn btn-secondary">
            <svg
              class="h-4 w-4 mr-2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
            Export
          </button>
        </div>

        <div class="flex flex-wrap gap-3">
          <div class="w-full sm:w-64">
            <ui-dropdown
              label="Filter"
              [options]="filterOptions"
              [value]="filter()"
              (valueChange)="filterChange.emit($event)"
            />
          </div>
          <div class="flex-1 min-w-[200px]">
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Date From</label>
            <input
              type="date"
              [ngModel]="dateFrom()"
              (ngModelChange)="dateFromChange.emit($event)"
              class="input w-full"
            />
          </div>
          <div class="flex-1 min-w-[200px]">
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Date To</label>
            <input
              type="date"
              [ngModel]="dateTo()"
              (ngModelChange)="dateToChange.emit($event)"
              class="input w-full"
            />
          </div>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
        <app-kpi-card
          icon="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
          label="Total Events"
          [value]="(trails()?.summary?.totalEvents || 0).toString()"
          color="primary"
        />
        <app-kpi-card
          icon="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
          label="User Actions"
          [value]="(trails()?.summary?.userActions || 0).toString()"
          color="primary"
        />
        <app-kpi-card
          icon="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
          label="Security Events"
          [value]="(trails()?.summary?.securityEvents || 0).toString()"
          color="warning"
        />
        <app-kpi-card
          icon="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
          label="System Events"
          [value]="(trails()?.summary?.systemEvents || 0).toString()"
          color="success"
        />
      </div>

      <div class="card overflow-hidden">
        <div class="overflow-x-auto">
          <table class="min-w-full divide-y divide-gray-200 dark:divide-dark-700">
            <thead class="bg-gray-50 dark:bg-dark-800">
              <tr>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Timestamp</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">User</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Action</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Resource</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Details</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">IP Address</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
              </tr>
            </thead>
            <tbody class="bg-white dark:bg-dark-800 divide-y divide-gray-200 dark:divide-dark-700">
              @for (event of trails()?.events; track event.id) {
                <tr class="hover:bg-gray-50 dark:hover:bg-dark-700">
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    {{ event.timestamp | date:'MMM d, y h:mm:ss a' }}
                  </td>
                  <td class="px-6 py-4 whitespace-nowrap">
                    <div class="text-sm font-medium text-gray-900 dark:text-white">{{ event.userName }}</div>
                    <div class="text-sm text-gray-500 dark:text-gray-400">{{ event.userRole }}</div>
                  </td>
                  <td class="px-6 py-4 whitespace-nowrap">
                    <span
                      [class]="actionTypeClass(event.actionType)"
                      class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium"
                    >
                      {{ event.actionType | titlecase }}
                    </span>
                  </td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    {{ event.resourceType }}: {{ event.resourceId }}
                  </td>
                  <td class="px-6 py-4 text-sm text-gray-500 dark:text-gray-400 max-w-xs truncate">
                    {{ event.details }}
                  </td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {{ event.ipAddress }}
                  </td>
                  <td class="px-6 py-4 whitespace-nowrap">
                    <span
                      [class]="eventStatusClass(event.status)"
                      class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium"
                    >
                      {{ event.status | titlecase }}
                    </span>
                  </td>
                </tr>
              }
            </tbody>
          </table>
        </div>

        <div class="bg-white dark:bg-dark-800 px-4 py-3 flex items-center justify-between border-t border-gray-200 dark:border-dark-700">
          <div class="flex-1 flex justify-between sm:hidden">
            <button class="btn btn-secondary">Previous</button>
            <button class="btn btn-secondary">Next</button>
          </div>
          <div class="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
            <div>
              <p class="text-sm text-gray-700 dark:text-gray-300">
                Showing <span class="font-medium">1</span> to <span class="font-medium">10</span> of
                <span class="font-medium">{{ trails()?.totalCount || 0 }}</span> results
              </p>
            </div>
            <div>
              <nav class="relative z-0 inline-flex rounded-md shadow-sm -space-x-px">
                <button class="btn btn-secondary rounded-l-md">Previous</button>
                <button class="btn btn-secondary">1</button>
                <button class="btn btn-primary">2</button>
                <button class="btn btn-secondary">3</button>
                <button class="btn btn-secondary rounded-r-md">Next</button>
              </nav>
            </div>
          </div>
        </div>
      </div>
    </div>
  `
})
export class AuditTrailsComponent {
  trails = input<AuditTrail | null>(null);
  filter = input<string>('all');
  dateFrom = input<string>('');
  dateTo = input<string>('');

  filterChange = output<string>();
  dateFromChange = output<string>();
  dateToChange = output<string>();
  export = output<void>();

  readonly filterOptions = [
    { value: 'all', label: 'All Activities' },
    { value: 'user', label: 'User Actions' },
    { value: 'system', label: 'System Events' },
    { value: 'security', label: 'Security Events' },
    { value: 'data', label: 'Data Changes' }
  ];

  protected actionTypeClass(actionType: string): string {
    switch (actionType) {
      case 'create':
        return 'bg-success-100 text-success-800';
      case 'update':
        return 'bg-primary-100 text-primary-800';
      case 'delete':
        return 'bg-error-100 text-error-800';
      case 'login':
        return 'bg-secondary-100 text-secondary-800';
      case 'logout':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }

  protected eventStatusClass(status: string): string {
    switch (status) {
      case 'success':
        return 'bg-success-100 text-success-800';
      case 'failed':
        return 'bg-error-100 text-error-800';
      case 'warning':
        return 'bg-warning-100 text-warning-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }
}
