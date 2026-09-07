import { defineStore } from 'pinia'
import type { SubmitTransportationRequestForm } from './submit-transportation-request.model'
import { submitTransportationRequest } from './submit-transportation-request-api'

// Fields the user has already visited, keyed the same way as validationErrors.
// A field's error message stays hidden until it is touched, so the form does not
// greet the user with a wall of red on a step they have not filled in yet.
interface TouchedFields {
  [field: string]: boolean;
}

export const useTransportationRequestStore = defineStore('transportationRequest', {
  state: () => ({
    currentStep: 1,
    totalSteps: 6,
    loading: false,
    success: false,
    error: null as string | null,
    requestNumber: '',
    // The furthest step reached so far. Steps behind it stay reachable through
    // the progress timeline, including the review step a user leaves through one
    // of its "Edit" shortcuts.
    furthestStep: 1,
    touchedFields: {} as TouchedFields,
    form: {
      serviceType: '',
      pickupLocation: {
        address: { street: '', city: '', country: '' },
        contactPerson: '',
        contactPhone: '',
        contactEmail: '',
        loadingType: 'DOCK',
        facilityType: 'WAREHOUSE',
        operatingHours: {}
      },
      deliveryLocation: {
        address: { street: '', city: '', country: '' },
        contactPerson: '',
        contactPhone: '',
        contactEmail: '',
        loadingType: 'DOCK',
        facilityType: 'WAREHOUSE',
        operatingHours: {}
      },
      cargo: {
        description: '',
        cargoType: 'GENERAL_CARGO',
        weight: 0,
        packaging: 'PALLETS',
        quantity: 1,
        unitType: 'pallets',
        value: 0,
        currency: 'EUR',
        fragile: false,
        stackable: true
      },
      requestedPickupDate: '',
      requestedDeliveryDate: '',
      specialInstructions: '',
      requiresInsurance: false,
      requiresCustomsClearance: false,
      priority: 'NORMAL',
      currency: 'EUR'
    } as SubmitTransportationRequestForm
  }),

  getters: {
    isFirstStep: (state) => state.currentStep === 1,
    isLastStep: (state) => state.currentStep === state.totalSteps,
    stepTitle: (state) => {
      const titles = [
        'Service Type',
        'Pickup Information',
        'Delivery Information',
        'Cargo Information',
        'Special Instructions',
        'Review & Submit'
      ]
      return titles[state.currentStep - 1]
    },
    stepDescription: (state) => {
      const descriptions = [
        'Select the type of transportation service you need',
        'Provide details about the pickup location',
        'Provide details about the delivery location',
        'Describe the cargo to be transported',
        'Add any special instructions or requirements',
        'Review your information and submit your request'
      ]
      return descriptions[state.currentStep - 1]
    },
    stepIcon: (state) => {
      const icons = [
        'TruckIcon',
        'MapPinIcon',
        'MapIcon',
        'CubeIcon',
        'DocumentTextIcon',
        'CheckIcon'
      ]
      return icons[state.currentStep - 1]
    },
    // Whether a given step has all of its required fields filled in
    isStepComplete: (state) => (step: number): boolean => {
      switch (step) {
        case 1: // Service Type
          return Boolean(state.form.serviceType)
        case 2: // Pickup Information
          return (
            Boolean(state.form.pickupLocation.address.street) &&
            Boolean(state.form.pickupLocation.address.city) &&
            Boolean(state.form.pickupLocation.address.country) &&
            Boolean(state.form.pickupLocation.contactPerson) &&
            Boolean(state.form.pickupLocation.contactPhone) &&
            Boolean(state.form.requestedPickupDate)
          )
        case 3: // Delivery Information
          return (
            Boolean(state.form.deliveryLocation.address.street) &&
            Boolean(state.form.deliveryLocation.address.city) &&
            Boolean(state.form.deliveryLocation.address.country) &&
            Boolean(state.form.deliveryLocation.contactPerson) &&
            Boolean(state.form.deliveryLocation.contactPhone)
          )
        case 4: // Cargo Information
          return (
            Boolean(state.form.cargo.description) &&
            state.form.cargo.weight > 0
          )
        case 5: // Special Instructions
          return true // No required fields
        case 6: // Review & Submit
          return true // Nothing of its own to fill in
        default:
          return false
      }
    },

    isStepValid(): boolean {
      return this.isStepComplete(this.currentStep)
    },

    /**
     * Whether a specific step can be opened from the progress timeline: it has
     * to have been reached before, and everything in front of it has to still be
     * complete - so leaving a step half-filled locks the ones behind it again.
     */
    canAccessStep(): (step: number) => boolean {
      return (step: number) => {
        if (step === 1) {
          return true
        }

        if (step > this.furthestStep) {
          return false
        }

        for (let earlier = 1; earlier < step; earlier++) {
          if (!this.isStepComplete(earlier)) {
            return false
          }
        }

        return true
      }
    },

    // Get validation errors for the current step
    validationErrors: (state) => {
      const errors: Record<string, string> = {};
      
      switch (state.currentStep) {
        case 1: // Service Type
          if (!state.form.serviceType) {
            errors.serviceType = 'Please select a service type';
          }
          break;
          
        case 2: // Pickup Information
          if (!state.form.pickupLocation.address.street) {
            errors['pickupLocation.address.street'] = 'Pickup address is required';
          }
          if (!state.form.pickupLocation.address.city) {
            errors['pickupLocation.address.city'] = 'City is required';
          }
          if (!state.form.pickupLocation.address.country) {
            errors['pickupLocation.address.country'] = 'Country is required';
          }
          if (!state.form.pickupLocation.contactPerson) {
            errors['pickupLocation.contactPerson'] = 'Contact person is required';
          }
          if (!state.form.pickupLocation.contactPhone) {
            errors['pickupLocation.contactPhone'] = 'Contact phone is required';
          }
          if (!state.form.requestedPickupDate) {
            errors.requestedPickupDate = 'Pickup date is required';
          }
          break;
          
        case 3: // Delivery Information
          if (!state.form.deliveryLocation.address.street) {
            errors['deliveryLocation.address.street'] = 'Delivery address is required';
          }
          if (!state.form.deliveryLocation.address.city) {
            errors['deliveryLocation.address.city'] = 'City is required';
          }
          if (!state.form.deliveryLocation.address.country) {
            errors['deliveryLocation.address.country'] = 'Country is required';
          }
          if (!state.form.deliveryLocation.contactPerson) {
            errors['deliveryLocation.contactPerson'] = 'Contact person is required';
          }
          if (!state.form.deliveryLocation.contactPhone) {
            errors['deliveryLocation.contactPhone'] = 'Contact phone is required';
          }
          break;
          
        case 4: // Cargo Information
          if (!state.form.cargo.description) {
            errors['cargo.description'] = 'Cargo description is required';
          }
          if (state.form.cargo.weight <= 0) {
            errors['cargo.weight'] = 'Weight must be greater than 0';
          }
          break;
      }

      return errors;
    },

    /**
     * The subset of validationErrors the form is allowed to show: a field only
     * complains once the user has actually been there. This is what the template
     * binds to - validationErrors itself stays the complete, untouched picture.
     */
    visibleValidationErrors(): Record<string, string> {
      const visible: Record<string, string> = {}

      for (const [field, message] of Object.entries(this.validationErrors)) {
        if (this.touchedFields[field]) {
          visible[field] = message
        }
      }

      return visible
    }
  },

  actions: {
    nextStep() {
      if (this.currentStep < this.totalSteps && this.isStepValid) {
        this.currentStep++
        this.furthestStep = Math.max(this.furthestStep, this.currentStep)
      }
    },

    validateCurrentStep() {
      // Return true if the step is valid, false otherwise
      return Object.keys(this.validationErrors).length === 0;
    },

    markTouched(field: string) {
      this.touchedFields[field] = true
    },

    /** Reveals every outstanding error of the current step at once. */
    touchCurrentStepFields() {
      for (const field of Object.keys(this.validationErrors)) {
        this.markTouched(field)
      }
    },

    prevStep() {
      if (this.currentStep > 1) {
        this.currentStep--
      }
    },
    goToStep(step: number) {
      if (step >= 1 && step <= this.totalSteps) {
        // Only allow navigation to accessible steps
        if (this.canAccessStep(step)) {
          this.currentStep = step
        }
      }
    },
    async submitRequest() {
      console.log('📝 Store submitRequest called')
      this.loading = true
      this.error = null
      
      try {
        console.log('📤 Calling submitTransportationRequest with form:', this.form)
        const response = await submitTransportationRequest(this.form)
        console.log('✅ submitTransportationRequest response:', response)
        this.success = true
        this.requestNumber = response.requestNumber
        console.log('✅ Store updated - success:', this.success, 'requestNumber:', this.requestNumber)
      } catch (error) {
        this.error = error instanceof Error ? error.message : 'Failed to submit request'
        console.error('❌ Submit transportation request failed:', error)
      } finally {
        this.loading = false
      }
    },
    resetForm() {
      this.currentStep = 1
      this.furthestStep = 1
      this.success = false
      this.error = null
      this.requestNumber = ''
      this.touchedFields = {}
      this.form = {
        serviceType: '',
        pickupLocation: {
          address: { street: '', city: '', country: '' },
          contactPerson: '',
          contactPhone: '',
          contactEmail: '',
          loadingType: 'DOCK',
          facilityType: 'WAREHOUSE',
          operatingHours: {}
        },
        deliveryLocation: {
          address: { street: '', city: '', country: '' },
          contactPerson: '',
          contactPhone: '',
          contactEmail: '',
          loadingType: 'DOCK',
          facilityType: 'WAREHOUSE',
          operatingHours: {}
        },
        cargo: {
          description: '',
          cargoType: 'GENERAL_CARGO',
          weight: 0,
          packaging: 'PALLETS',
          quantity: 1,
          unitType: 'pallets',
          value: 0,
          currency: 'EUR',
          fragile: false,
          stackable: true
        },
        requestedPickupDate: '',
        requestedDeliveryDate: '',
        specialInstructions: '',
        requiresInsurance: false,
        requiresCustomsClearance: false,
        priority: 'NORMAL',
        currency: 'EUR'
      }
    }
  }
})