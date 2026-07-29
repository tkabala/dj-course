import { Component, input, output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DropdownComponent } from '../../../ui-library/Dropdown.component';
import { Heading3Component, Heading4Component } from '../../../ui-library/Typography/Typography.component';
import { FinancialReport } from '../reports.model';
import { KpiCardComponent } from './kpi-card.component';

@Component({
  selector: 'app-financial-reports',
  standalone: true,
  imports: [
    CommonModule,
    DropdownComponent,
    Heading3Component,
    Heading4Component,
    KpiCardComponent
  ],
  template: `
    <div class="p-6">
      <div class="flex justify-between items-center mb-6">
        <ui-heading3>Financial Reports</ui-heading3>
        <div class="flex space-x-3">
          <ui-dropdown
            label="Period"
            [options]="periodOptions"
            [value]="period()"
            (valueChange)="periodChange.emit($event)"
          />
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
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <app-kpi-card
          icon="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1"
          label="Total Revenue"
          [value]="'$' + formatCurrency(report()?.totalRevenue)"
          [sublabel]="'+' + (report()?.revenueGrowth || 0) + '% vs last period'"
          color="success"
        />
        <app-kpi-card
          icon="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"
          label="Operating Costs"
          [value]="'$' + formatCurrency(report()?.operatingCosts)"
          [sublabel]="'+' + (report()?.costIncrease || 0) + '% vs last period'"
          color="primary"
        />
        <app-kpi-card
          icon="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
          label="Net Profit"
          [value]="'$' + formatCurrency(report()?.netProfit)"
          [sublabel]="(report()?.profitMargin || 0) + '% margin'"
          color="warning"
        />
        <app-kpi-card
          icon="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z"
          label="Outstanding Invoices"
          [value]="'$' + formatCurrency(report()?.outstandingInvoices)"
          [sublabel]="(report()?.overdueCount || 0) + ' overdue'"
          color="secondary"
        />
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="card p-6">
          <ui-heading4>Revenue by Service Type</ui-heading4>
          <div class="space-y-4 mt-4">
            @for (service of report()?.revenueByService; track service.serviceName) {
              <div class="flex items-center justify-between">
                <div class="flex items-center">
                  <div class="w-3 h-3 rounded-full mr-3" [style.background-color]="service.color"></div>
                  <span class="text-sm text-gray-600 dark:text-gray-400">{{ service.serviceName }}</span>
                </div>
                <div class="text-right">
                  <div class="text-sm font-medium text-gray-900 dark:text-white">\${{ formatCurrency(service.revenue) }}</div>
                  <div class="text-xs text-gray-500">{{ service.percentage }}%</div>
                </div>
              </div>
            }
          </div>
        </div>

        <div class="card p-6">
          <ui-heading4>Monthly Billing Summary</ui-heading4>
          <div class="overflow-x-auto mt-4">
            <table class="min-w-full">
              <thead>
                <tr class="border-b border-gray-200 dark:border-dark-600">
                  <th class="text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider py-2">Contractor</th>
                  <th class="text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider py-2">Amount</th>
                  <th class="text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider py-2">Status</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-gray-200 dark:divide-dark-600">
                @for (billing of report()?.billingDetails; track billing.contractorName) {
                  <tr class="py-2">
                    <td class="text-sm text-gray-900 dark:text-white py-2">{{ billing.contractorName }}</td>
                    <td class="text-sm text-gray-900 dark:text-white text-right py-2">\${{ formatCurrency(billing.amount) }}</td>
                    <td class="text-right py-2">
                      <span
                        [class]="billingStatusClass(billing.status)"
                        class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium"
                      >
                        {{ billing.status | titlecase }}
                      </span>
                    </td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  `
})
export class FinancialReportsComponent {
  report = input<FinancialReport | null>(null);
  period = input<string>('month');
  periodChange = output<string>();
  export = output<void>();

  readonly periodOptions = [
    { value: 'month', label: 'This Month' },
    { value: 'quarter', label: 'This Quarter' },
    { value: 'year', label: 'This Year' },
    { value: 'custom', label: 'Custom Range' }
  ];

  protected formatCurrency(value: number | undefined): string {
    if (!value) return '0';
    return new Intl.NumberFormat('en-US').format(value);
  }

  protected billingStatusClass(status: string): string {
    switch (status) {
      case 'paid':
        return 'bg-success-100 text-success-800';
      case 'pending':
        return 'bg-warning-100 text-warning-800';
      case 'overdue':
        return 'bg-error-100 text-error-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }
}
