import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReportsPageState } from './reports-page.state';
import { ReportsHeaderComponent } from './components/reports-header.component';
import { ReportTabsComponent } from './components/report-tabs.component';
import { OperationalMetricsComponent } from './components/operational-metrics.component';
import { UtilizationReportsComponent } from './components/utilization-reports.component';
import { FinancialReportsComponent } from './components/financial-reports.component';
import { AuditTrailsComponent } from './components/audit-trails.component';

@Component({
  selector: 'app-reports-page',
  standalone: true,
  imports: [
    CommonModule,
    ReportsHeaderComponent,
    ReportTabsComponent,
    OperationalMetricsComponent,
    UtilizationReportsComponent,
    FinancialReportsComponent,
    AuditTrailsComponent
  ],
  providers: [ReportsPageState],
  template: `
    <div class="space-y-6">
      <app-reports-header />

      <app-report-tabs
        [tabs]="state.reportTabs"
        [activeTab]="state.activeTab()"
        (tabChange)="state.setActiveTab($event)"
      />

      @if (state.activeTab() === 'operational') {
        <app-operational-metrics
          [metrics]="state.operationalMetrics()"
          [period]="state.operationalPeriod()"
          (periodChange)="state.operationalPeriod.set($event); state.loadOperationalMetrics()"
          (export)="state.exportReport('operational')"
        />
      }

      @if (state.activeTab() === 'utilization') {
        <app-utilization-reports
          [report]="state.utilizationReport()"
          [period]="state.utilizationPeriod()"
          (periodChange)="state.utilizationPeriod.set($event); state.loadUtilizationReports()"
          (export)="state.exportReport('utilization')"
        />
      }

      @if (state.activeTab() === 'financial') {
        <app-financial-reports
          [report]="state.financialReport()"
          [period]="state.financialPeriod()"
          (periodChange)="state.financialPeriod.set($event); state.loadFinancialReports()"
          (export)="state.exportReport('financial')"
        />
      }

      @if (state.activeTab() === 'audit') {
        <app-audit-trails
          [trails]="state.auditTrails()"
          [filter]="state.auditFilter()"
          [dateFrom]="state.auditDateFrom()"
          [dateTo]="state.auditDateTo()"
          (filterChange)="state.auditFilter.set($event); state.loadAuditTrails()"
          (dateFromChange)="state.auditDateFrom.set($event); state.loadAuditTrails()"
          (dateToChange)="state.auditDateTo.set($event); state.loadAuditTrails()"
          (export)="state.exportReport('audit')"
        />
      }
    </div>
  `
})
export class ReportsPageComponent implements OnInit {
  protected readonly state = inject(ReportsPageState);

  ngOnInit(): void {
    this.state.loadAll();
  }
}
